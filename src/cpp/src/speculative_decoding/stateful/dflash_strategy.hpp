// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <openvino/genai/perf_metrics.hpp>
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>

#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "stateful_pipeline_base.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// ============================================================================
// DFlash Speculative Decoding Strategy
//
// DFlash is a block-diffusion speculative decoding method.  Unlike Eagle3's
// autoregressive draft loop, DFlash generates an entire block of `block_size`
// candidate tokens in a SINGLE non-causal (bidirectional) draft pass.
//
// Key differences from Eagle3:
//   1. Draft model uses NON-CAUSAL cross-attention (bidirectional).
//   2. Generates block_size tokens in ONE pass (parallel, not autoregressive).
//   3. Extracts hidden states from 5 target layers [1,9,17,25,33].
//   4. Uses mask_token_id (151669) to fill unknown positions.
//   5. Draft model embeds its own input_ids (shared embed_tokens) and produces
//      logits (shared lm_head).  FC projection stays inside draft model.
//   6. Context feature = concat of 5 hidden states → [batch, seq, 5*hidden_size].
//   7. No d2t mapping table (draft vocab == target vocab).
// ============================================================================

/// @brief DFlash model inference output
struct DFlashInferenceOutput {
    ov::Tensor logits;           ///< Output logits [batch, seq_len, vocab_size]
    ov::Tensor hidden_features;  ///< Hidden states for accumulation
};

/// @brief Context for a single DFlash forward pass
struct DFlashInferContext {
    size_t input_token_count = 0;             ///< Number of input tokens for this inference
    size_t sample_count = 1;                  ///< Number of positions to sample from
    size_t num_tokens_to_validate = 0;        ///< Number of draft tokens to validate (target only)
};

/// @brief Result from a DFlash forward pass
struct DFlashInferResult {
    DFlashInferenceOutput output;         ///< Raw model outputs
    std::vector<int64_t> sampled_tokens;  ///< Sampled token(s)
};

/**
 * @brief Base class for DFlash model inference wrappers
 *
 * Provides shared functionality for target and draft model wrappers including
 * sequence management, KV cache operations, tensor building, and token sampling.
 *
 * DFlash target model is stateful (KV cache) just like Eagle3.
 * DFlash draft model is STATELESS — no KV cache, no autoregressive state.
 */
class DFlashInferWrapperBase {
public:
    explicit DFlashInferWrapperBase(const ov::genai::ModelDesc& model_desc);
    virtual ~DFlashInferWrapperBase() = default;

    std::string device() const {
        return m_device;
    }

    void append_tokens(const std::vector<int64_t>& tokens);
    void truncate_sequence(size_t size);
    void trim_kv_cache(size_t tokens_to_remove);
    void reset_state();
    void release_memory();

    /// @brief Returns total sequence length (prompt + generated)
    size_t get_sequence_length() const {
        if (auto seq = get_sequence(0)) {
            return m_sequence_group->get_prompt_len() + seq->get_generated_len();
        }
        return 0;
    }

    /// @brief Returns only generated tokens
    const std::vector<int64_t>& get_generated_tokens() const {
        static const std::vector<int64_t> empty;
        if (auto seq = get_sequence(0)) {
            return seq->get_generated_ids();
        }
        return empty;
    }

    SequenceGroup::Ptr get_sequence_group() const {
        return m_sequence_group;
    }

    Sequence::Ptr get_sequence(size_t index) const {
        if (m_sequence_group) {
            const auto& sequences = m_sequence_group->get_sequences();
            if (index < sequences.size()) {
                return sequences[index];
            }
        }
        return nullptr;
    }

    Sequence::Ptr get_current_sequence() const {
        return get_sequence(0);
    }

    ov::Tensor get_logits() const;
    ov::Tensor get_hidden_features() const;

    void build_model_inputs(const size_t token_count,
                            ov::Tensor& input_ids,
                            ov::Tensor& attention_mask,
                            ov::Tensor& position_ids);

    std::vector<int64_t> sample_tokens(const ov::Tensor& logits,
                                       size_t input_token_count,
                                       size_t sample_count,
                                       size_t num_tokens_to_validate = 0);

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

protected:
    static constexpr size_t BATCH_SIZE = 1;

    uint64_t execute_inference();
    void update_inference_time(uint64_t inference_time_us);
    void record_generated_tokens(size_t actual_generated_count);

    std::string m_device;
    ov::AnyMap m_properties;
    ov::genai::Tokenizer m_tokenizer;
    mutable ov::InferRequest m_request;
    ov::genai::utils::KVAxesPosition m_kv_axes_pos;
    size_t m_max_prompt_len = 0;

    SequenceGroup::Ptr m_sequence_group;
    Sampler m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    ov::genai::utils::CacheTypes m_cache_types;
};

/**
 * @brief Target model wrapper for DFlash
 *
 * Stateful model with KV cache.  Validates draft predictions, generates
 * final output, and produces hidden states from 5 layers concatenated
 * along the last dimension → [batch, seq, 5*hidden_size].
 */
class DFlashTargetWrapper : public DFlashInferWrapperBase {
public:
    explicit DFlashTargetWrapper(const ov::genai::ModelDesc& model_desc);
    ~DFlashTargetWrapper() = default;

    /// @brief Initializes sequence with prompt tokens
    void initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config);

    DFlashInferenceOutput infer(const ov::Tensor& input_ids,
                                const ov::Tensor& attention_mask,
                                const ov::Tensor& position_ids);

    /// @brief Forward: build inputs → infer → sample → store hidden states
    DFlashInferResult forward(const DFlashInferContext& ctx);
};

/**
 * @brief Draft model wrapper for DFlash
 *
 * STATELESS model — no KV cache.  Takes token IDs (with mask tokens)
 * and accumulated target hidden states as cross-attention context.
 * Generates an entire block of logits in a single non-causal pass.
 *
 * Draft model inputs:
 *   - input_ids:      [1, block_size]  — [last_token, mask, mask, ..., mask]
 *   - hidden_states:  [1, ctx_len, 5*hidden_size]  — accumulated target features
 *   - position_ids:   [1, ctx_len + block_size]  — full position range
 *
 * Draft model output:
 *   - logits:         [1, block_size, vocab_size]
 */
class DFlashDraftWrapper {
public:
    explicit DFlashDraftWrapper(const ov::genai::ModelDesc& model_desc);
    ~DFlashDraftWrapper() = default;

    std::string device() const { return m_device; }

    void release_memory();
    void reset_perf_metrics();

    /// @brief Run a single non-causal draft pass, returns logits for the whole block
    ov::Tensor infer(const ov::Tensor& input_ids,
                     const ov::Tensor& hidden_states,
                     const ov::Tensor& position_ids);

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

private:
    std::string m_device;
    ov::AnyMap m_properties;
    mutable ov::InferRequest m_request;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
};

/**
 * @brief Stateful DFlash speculative decoding pipeline
 *
 * Implements the DFlash algorithm:
 *   1. Prefill: target processes prompt → first token + hidden states
 *   2. Draft: single non-causal pass produces block_size candidate tokens
 *   3. Verify: target validates candidates in parallel
 *   4. Accept matching prefix, trim KV cache for rejected tail
 *   5. Accumulate accepted hidden states for next draft pass
 *   6. Repeat until EOS or max_new_tokens
 *
 * Unlike Eagle3 which runs multiple autoregressive draft iterations,
 * DFlash always runs exactly ONE draft inference per speculation round.
 */
class StatefulDFlashLLMPipeline : public StatefulSpeculativePipelineBase {
public:
    StatefulDFlashLLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                              const ov::genai::ModelDesc& draft_model_desc);
    ~StatefulDFlashLLMPipeline();

    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;

    void finish_chat() override;

protected:
    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config) override;

    EncodedResults generate_tokens(const EncodedInputs& inputs,
                                   const GenerationConfig& config,
                                   StreamerVariant streamer) override;

private:
    struct SpeculativeResult {
        size_t accepted_tokens_count = 0;
        bool eos_reached = false;
        std::vector<int64_t> validated_tokens;
    };

    /// @brief Build the mask-token block: [last_token, mask, mask, ..., mask]
    ov::Tensor build_draft_input_ids(int64_t last_token) const;

    /// @brief Build position_ids covering [0 .. ctx_len + block_size - 1]
    ov::Tensor build_draft_position_ids(size_t ctx_len) const;

    /// @brief Concatenate new hidden states (accepted only) to accumulated context
    void accumulate_hidden_states(const ov::Tensor& new_hidden, size_t num_accepted);

    SpeculativeResult run_speculative_iteration(int64_t eos_token_id,
                                                size_t current_generated_tokens,
                                                size_t max_new_tokens);

    std::unique_ptr<DFlashDraftWrapper> m_draft;
    std::unique_ptr<DFlashTargetWrapper> m_target;

    size_t m_block_size = 8;                              ///< Draft block size (default 8)
    int64_t m_mask_token_id = 151669;                     ///< Mask token for unknown positions
    std::vector<int32_t> m_target_layer_ids;              ///< Target layers for hidden extraction
    size_t m_prompt_length = 0;

    /// @brief Accumulated hidden states from target model [1, ctx_len, 5*hidden_size]
    /// Grows as tokens are accepted. Used as cross-attention context for draft model.
    ov::Tensor m_accumulated_hidden;
};

// ============================================================================
// DFlash model transform utilities (analogous to eagle3_model_transforms.hpp)
// ============================================================================
namespace utils {
namespace dflash {

/// @brief Runtime configuration for DFlash speculative decoding
struct DFlashRTInfo {
    bool dflash_mode = false;                  ///< Enable DFlash mode
    int32_t block_size = 8;                    ///< Draft block size
    int64_t mask_token_id = 151669;            ///< Mask token for unknown positions
    std::vector<int32_t> target_layer_ids;     ///< Target layers for hidden extraction
};

/// @brief Extracts DFlash configuration from model config
DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config);

/// @brief Applies DFlash runtime info from model to properties map
void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties);

/// @brief Transform target model to output concatenated hidden states from specified layers
/// Adds "last_hidden_state" result: Concat of 5 residual Add outputs → [batch, seq, 5*hidden_size]
void transform_target_hidden_state(std::shared_ptr<ov::Model>& model,
                                   const std::vector<int32_t>& target_layer_ids);

}  // namespace dflash
}  // namespace utils

}  // namespace genai
}  // namespace ov
