// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>

#include "continuous_batching/timer.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/result.hpp"
#include "utils.hpp"

// ============================================================================
// DFlash Speculative Decoding — C++ Implementation
//
// Reference: dflash_pipeline_v5.py (Python prototype)
// Pattern: Eagle3 strategy (eagle3_strategy.cpp)
//
// DFlash key characteristics:
//   - Draft model is STATELESS (no KV cache) — runs one non-causal pass per block
//   - Target model is STATEFUL (KV cache) — identical to Eagle3 target
//   - Hidden states from 5 target layers are concatenated and accumulated
//   - Draft input is [last_token, mask, mask, ..., mask] (block_size tokens)
//   - FC projection lives inside draft model (not moved to target like Eagle3)
// ============================================================================


namespace {

ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

/// @brief Concatenate two 3D tensors along dimension 1 (sequence length)
/// @param a Tensor [1, seq_a, hidden]
/// @param b Tensor [1, seq_b, hidden]
/// @return New tensor [1, seq_a + seq_b, hidden]
ov::Tensor concat_along_seq(const ov::Tensor& a, const ov::Tensor& b) {
    const auto a_shape = a.get_shape();
    const auto b_shape = b.get_shape();

    OPENVINO_ASSERT(a_shape.size() == 3 && b_shape.size() == 3, "Expected 3D tensors");
    OPENVINO_ASSERT(a_shape[0] == b_shape[0] && a_shape[2] == b_shape[2],
                    "Batch and hidden dimensions must match");

    const size_t batch = a_shape[0];
    const size_t seq_a = a_shape[1];
    const size_t seq_b = b_shape[1];
    const size_t hidden = a_shape[2];
    const size_t total_seq = seq_a + seq_b;

    ov::Tensor result(a.get_element_type(), {batch, total_seq, hidden});

    const size_t elem_size = a.get_byte_size() / a.get_size();
    const auto* a_ptr = static_cast<const uint8_t*>(a.data());
    const auto* b_ptr = static_cast<const uint8_t*>(b.data());
    auto* r_ptr = static_cast<uint8_t*>(result.data());

    // Copy a's data
    const size_t a_bytes = seq_a * hidden * elem_size;
    std::memcpy(r_ptr, a_ptr, a_bytes);

    // Copy b's data
    const size_t b_bytes = seq_b * hidden * elem_size;
    std::memcpy(r_ptr + a_bytes, b_ptr, b_bytes);

    return result;
}

/// @brief Slice first `count` positions from dim=1 of a 3D tensor
ov::Tensor slice_seq(const ov::Tensor& tensor, size_t start, size_t count) {
    const auto shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && start + count <= shape[1], "Slice out of bounds");

    auto [start_coord, end_coord] = ov::genai::utils::make_roi(shape, 1, start, start + count);
    return ov::Tensor(tensor, start_coord, end_coord);
}

}  // anonymous namespace

namespace ov::genai {

// ============================================================================
// DFlashInferWrapperBase — shared target/draft functionality
// ============================================================================

DFlashInferWrapperBase::DFlashInferWrapperBase(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_tokenizer(model_desc.tokenizer),
      m_sampler(model_desc.tokenizer) {
    m_kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);

    m_cache_types = utils::get_cache_types(*model_desc.model);
    OPENVINO_ASSERT(!m_cache_types.has_linear(),
        "Stateful speculative decoding does not support models with linear attention states.");

    if (m_device == "NPU") {
        auto [compiled, kv_desc] = utils::compile_decoder_for_npu(model_desc.model, m_properties, m_kv_axes_pos);
        m_max_prompt_len = kv_desc.max_prompt_len;
        m_request = compiled.create_infer_request();
    } else {
        m_request =
            utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();
    }

    // Initialize performance metrics
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};

    m_sequence_group = nullptr;
}

void DFlashInferWrapperBase::append_tokens(const std::vector<int64_t>& tokens) {
    if (tokens.empty())
        return;
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");
    for (auto token : tokens) {
        current_sequence->append_token(token, 0.0f);
    }
}

void DFlashInferWrapperBase::truncate_sequence(size_t size) {
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const size_t prompt_len = m_sequence_group->get_prompt_len();
    const size_t current_len = prompt_len + current_sequence->get_generated_len();

    if (size < current_len) {
        OPENVINO_ASSERT(size >= prompt_len, "Cannot truncate prompt tokens");
        const size_t tokens_to_remove = current_len - size;
        current_sequence->remove_last_tokens(tokens_to_remove);
    }
}

void DFlashInferWrapperBase::trim_kv_cache(size_t tokens_to_remove) {
    const size_t current_len = get_sequence_length();
    if (tokens_to_remove == 0 || current_len == 0) {
        return;
    }

    OPENVINO_ASSERT(tokens_to_remove > 0 && tokens_to_remove < current_len,
                    "Cannot trim ", tokens_to_remove, " tokens from ", current_len, " tokens");

    if (m_device != "NPU") {
        utils::CacheState state(m_cache_types);
        state.num_tokens_to_trim = tokens_to_remove;
        state.seq_length_axis = m_kv_axes_pos.seq_len;
        state.reset_mem_state = false;
        utils::trim_kv_cache(m_request, state, {});
    }
}

void DFlashInferWrapperBase::reset_state() {
    m_sequence_group = nullptr;
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

void DFlashInferWrapperBase::release_memory() {
    m_request.get_compiled_model().release_memory();
}

void DFlashInferWrapperBase::build_model_inputs(const size_t input_token_count,
                                                ov::Tensor& input_ids,
                                                ov::Tensor& attention_mask,
                                                ov::Tensor& position_ids) {
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const auto& generated_ids = current_sequence->get_generated_ids();

    const size_t prompt_len = prompt_ids.size();
    const size_t generated_len = generated_ids.size();
    const size_t total_len = prompt_len + generated_len;
    const size_t start_pos = total_len - input_token_count;

    OPENVINO_ASSERT(input_token_count > 0 && input_token_count <= total_len,
                    "Invalid input_token_count: ", input_token_count, ", total_len: ", total_len);

    // Allocate tensors
    input_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    position_ids = ov::Tensor(ov::element::i64, {1, input_token_count});

    int64_t* input_ids_ptr = input_ids.data<int64_t>();
    int64_t* position_ids_ptr = position_ids.data<int64_t>();

    // Fill input_ids and position_ids from sequence
    if (start_pos < prompt_len) {
        const size_t prompt_count = std::min(input_token_count, prompt_len - start_pos);
        std::copy_n(prompt_ids.data() + start_pos, prompt_count, input_ids_ptr);
        std::iota(position_ids_ptr, position_ids_ptr + prompt_count, static_cast<int64_t>(start_pos));

        if (input_token_count > prompt_count) {
            const size_t generated_count = input_token_count - prompt_count;
            std::copy_n(generated_ids.data(), generated_count, input_ids_ptr + prompt_count);
            std::iota(position_ids_ptr + prompt_count,
                      position_ids_ptr + prompt_count + generated_count,
                      static_cast<int64_t>(prompt_len));
        }
    } else {
        const size_t generated_start = start_pos - prompt_len;
        std::copy_n(generated_ids.data() + generated_start, input_token_count, input_ids_ptr);
        std::iota(position_ids_ptr,
                  position_ids_ptr + input_token_count,
                  static_cast<int64_t>(prompt_len + generated_start));
    }

    // Build attention mask
    const size_t attention_mask_len = static_cast<size_t>(position_ids_ptr[input_token_count - 1] + 1);
    attention_mask = ov::Tensor(ov::element::i64, {1, attention_mask_len});
    std::fill_n(attention_mask.data<int64_t>(), attention_mask_len, 1);
}

std::vector<int64_t> DFlashInferWrapperBase::sample_tokens(const ov::Tensor& logits,
                                                           size_t input_token_count,
                                                           size_t sample_count,
                                                           size_t num_tokens_to_validate) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid logits shape");
    OPENVINO_ASSERT(sample_count > 0 && sample_count <= shape[1],
                    "Invalid sample_count: ", sample_count, ", logits seq_len: ", shape[1]);

    const bool is_validation_mode = num_tokens_to_validate > 0;

    auto sequence_group = get_sequence_group();
    OPENVINO_ASSERT(sequence_group, "SequenceGroup not initialized");

    auto current_seq = get_current_sequence();
    OPENVINO_ASSERT(current_seq, "No sequence at index 0");

    const size_t prev_generated_len = current_seq->get_generated_len();
    const size_t logits_seq_len = shape[1];

    // Slice logits to last 'sample_count' positions if needed
    ov::Tensor sliced_logits = logits;
    if (sample_count < logits_seq_len) {
        auto [start_coord, end_coord] =
            ov::genai::utils::make_roi(shape, 1, logits_seq_len - sample_count, logits_seq_len);
        sliced_logits = ov::Tensor(logits, start_coord, end_coord);
    }

    // Configure sequence group for sampling
    sequence_group->schedule_tokens(input_token_count);
    sequence_group->set_output_seq_len(sample_count);
    sequence_group->set_num_validated_tokens(num_tokens_to_validate);

    // Execute sampling
    m_sampler.sample({sequence_group}, sliced_logits, is_validation_mode);
    sequence_group->finish_iteration();

    // Extract results
    const auto& generated_ids = current_seq->get_generated_ids();
    const size_t new_generated_len = generated_ids.size();

    if (!is_validation_mode) {
        OPENVINO_ASSERT(new_generated_len - prev_generated_len == sample_count,
                        "Sampled token count mismatch");

        std::vector<int64_t> result_tokens(generated_ids.end() - sample_count, generated_ids.end());
        record_generated_tokens(sample_count);
        return result_tokens;
    } else {
        const size_t result_count = new_generated_len - prev_generated_len + num_tokens_to_validate;
        std::vector<int64_t> result_tokens(generated_ids.end() - result_count, generated_ids.end());
        record_generated_tokens(result_tokens.size());
        return result_tokens;
    }
}

ov::Tensor DFlashInferWrapperBase::get_logits() const {
    return m_request.get_tensor("logits");
}

ov::Tensor DFlashInferWrapperBase::get_hidden_features() const {
    // DFlash target model outputs "last_hidden_state" which is the concatenation
    // of 5 layer hidden states along dim=-1 → [1, seq_len, 5*hidden_size]
    auto hidden_state = m_request.get_tensor("last_hidden_state");
    const auto shape = hidden_state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1, "Invalid hidden state shape");

    const size_t output_seq_len = shape[1];
    const size_t actual_seq_len = m_request.get_tensor("input_ids").get_shape()[1];

    if (output_seq_len == actual_seq_len)
        return hidden_state;

    OPENVINO_ASSERT(actual_seq_len <= output_seq_len,
                    "Sequence length mismatch: actual=", actual_seq_len, ", output=", output_seq_len);
    auto [start_coord, end_coord] =
        ov::genai::utils::make_roi(shape, 1, output_seq_len - actual_seq_len, output_seq_len);
    return ov::Tensor(hidden_state, start_coord, end_coord);
}

uint64_t DFlashInferWrapperBase::execute_inference() {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    return duration_us;
}

void DFlashInferWrapperBase::update_inference_time(uint64_t inference_time_us) {
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
}

void DFlashInferWrapperBase::record_generated_tokens(size_t actual_generated_count) {
    m_raw_perf_metrics.m_batch_sizes.emplace_back(actual_generated_count);
}

// ============================================================================
// DFlashTargetWrapper — stateful target model (same pattern as Eagle3)
// ============================================================================

DFlashTargetWrapper::DFlashTargetWrapper(const ov::genai::ModelDesc& model_desc)
    : DFlashInferWrapperBase(model_desc) {}

void DFlashTargetWrapper::initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "Expected input_ids shape [1, seq_len]");

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t seq_len = shape[1];
    OPENVINO_ASSERT(seq_len > 0, "Empty prompt");

    TokenIds prompt_ids(ids_data, ids_data + seq_len);
    m_sequence_group = std::make_shared<SequenceGroup>(0, prompt_ids, config, 0);
    OPENVINO_ASSERT(m_sequence_group->num_total_seqs() == 1);
}

DFlashInferenceOutput DFlashTargetWrapper::infer(const ov::Tensor& input_ids,
                                                 const ov::Tensor& attention_mask,
                                                 const ov::Tensor& position_ids) {
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);

    uint64_t time_us = execute_inference();
    update_inference_time(time_us);

    DFlashInferenceOutput output;
    output.logits = get_logits();
    output.hidden_features = get_hidden_features();
    return output;
}

DFlashInferResult DFlashTargetWrapper::forward(const DFlashInferContext& ctx) {
    // 1. Build inputs from sequence state
    ov::Tensor input_ids, attention_mask, position_ids;
    build_model_inputs(ctx.input_token_count, input_ids, attention_mask, position_ids);

    // 2. Infer
    auto output = infer(input_ids, attention_mask, position_ids);

    // 3. Sample
    auto sampled = sample_tokens(output.logits, ctx.input_token_count, ctx.sample_count, ctx.num_tokens_to_validate);

    return DFlashInferResult{std::move(output), std::move(sampled)};
}

// ============================================================================
// DFlashDraftWrapper — stateless draft model (no KV cache)
// ============================================================================

DFlashDraftWrapper::DFlashDraftWrapper(const ov::genai::ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties) {
    // Draft model is stateless — simply compile and create infer request
    // Inputs: input_ids [1, block_size], hidden_states [1, ctx_len, 5*hidden_size],
    //         position_ids [1, ctx_len + block_size]
    // Output: logits [1, block_size, vocab_size]
    m_request =
        utils::singleton_core().compile_model(model_desc.model, m_device, m_properties).create_infer_request();

    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
}

void DFlashDraftWrapper::release_memory() {
    m_request.get_compiled_model().release_memory();
}

void DFlashDraftWrapper::reset_perf_metrics() {
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

ov::Tensor DFlashDraftWrapper::infer(const ov::Tensor& input_ids,
                                     const ov::Tensor& hidden_states,
                                     const ov::Tensor& position_ids) {
    // DFlash draft model inputs:
    //   input(0) = input_ids:     [1, block_size]        — token IDs with mask tokens
    //   input(1) = hidden_states: [1, ctx_len, 5*hidden] — target hidden context
    //   input(2) = position_ids:  [1, ctx_len + block_size] — full position range
    //
    // The draft model internally:
    //   1. embed_tokens(input_ids) → noise embedding [1, block_size, hidden_size]
    //   2. FC projects hidden_states from 5*hidden → hidden_size
    //   3. Non-causal cross-attention: query=noise, key/value=concat(projected_ctx, noise)
    //   4. lm_head(output) → logits [1, block_size, vocab_size]
    //
    // Position IDs must cover the full range [0..ctx_len+block_size-1] because
    // rotary embeddings are applied to the concatenated K sequence.

    OPENVINO_ASSERT(input_ids.get_shape().size() == 2, "input_ids must be 2D");
    OPENVINO_ASSERT(hidden_states.get_shape().size() == 3, "hidden_states must be 3D");
    OPENVINO_ASSERT(position_ids.get_shape().size() == 2, "position_ids must be 2D");

    // Set inputs by index (draft model uses positional input names)
    m_request.set_input_tensor(0, input_ids);
    m_request.set_input_tensor(1, hidden_states);
    m_request.set_input_tensor(2, position_ids);

    // Execute inference
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();

    // Update metrics
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(duration_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(duration_us));
    m_raw_perf_metrics.m_batch_sizes.emplace_back(input_ids.get_shape()[1]);

    // Return logits [1, block_size, vocab_size]
    return m_request.get_output_tensor(0);
}

// ============================================================================
// StatefulDFlashLLMPipeline — main pipeline
// ============================================================================

StatefulDFlashLLMPipeline::StatefulDFlashLLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                                                     const ov::genai::ModelDesc& draft_model_desc)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config) {

    // Extract DFlash configuration from draft model properties
    // Expected properties: block_size, mask_token_id, target_layer_ids
    auto it_block = draft_model_desc.properties.find("block_size");
    if (it_block != draft_model_desc.properties.end()) {
        m_block_size = it_block->second.as<size_t>();
    }

    auto it_mask = draft_model_desc.properties.find("mask_token_id");
    if (it_mask != draft_model_desc.properties.end()) {
        m_mask_token_id = it_mask->second.as<int64_t>();
    }

    OPENVINO_ASSERT(draft_model_desc.properties.find("target_layer_ids") != draft_model_desc.properties.end(),
                    "target_layer_ids must be present in draft model properties");
    m_target_layer_ids = draft_model_desc.properties.at("target_layer_ids").as<std::vector<int32_t>>();

    OPENVINO_ASSERT(m_target_layer_ids.size() == 5,
                    "DFlash requires exactly 5 target layers for hidden state extraction, got: ",
                    m_target_layer_ids.size());

    auto target_model = target_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(target_model, "Target model must not be null");
    OPENVINO_ASSERT(draft_model, "Draft model must not be null");

    // ---- Model preparation ----
    // Unlike Eagle3, DFlash does NOT:
    //   - Move FC from draft to main (FC stays in draft model)
    //   - Share vocabulary at graph level (draft has its own embed_tokens + lm_head)
    //   - Extract d2t mapping table (draft vocab == target vocab)
    //
    // What we DO:
    //   - Transform target model to output hidden states from 5 layers concatenated
    //     along dim=-1 → [batch, seq, 5*hidden_size]

    // Add hidden state extraction to target model
    // This adds a "last_hidden_state" output that is the concat of residual Add
    // outputs from layers [1, 9, 17, 25, 33]
    utils::dflash::transform_target_hidden_state(target_model, m_target_layer_ids);

    // Configure and create target model
    auto target_desc = target_model_desc;
    m_target = std::make_unique<DFlashTargetWrapper>(target_desc);

    // Configure and create draft model
    // NOTE: Draft model benefits from f32 inference precision on GPU to avoid FP16 overflow
    // in FC layer and attention with extreme hidden state values.
    // The f32 precision should be configured in the draft model's properties (ModelDesc)
    // by the caller or via INFERENCE_PRECISION_HINT when constructing the pipeline.
    auto draft_desc = draft_model_desc;
    m_draft = std::make_unique<DFlashDraftWrapper>(draft_desc);
}

StatefulDFlashLLMPipeline::~StatefulDFlashLLMPipeline() {
    m_target->release_memory();
    m_draft->release_memory();
}

GenerationConfig StatefulDFlashLLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);
    // DFlash uses block_size from config, not num_assistant_tokens
    // but we can optionally override block_size from generation config
    if (config.num_assistant_tokens > 0) {
        m_block_size = config.num_assistant_tokens;
    }
    return config;
}

ov::Tensor StatefulDFlashLLMPipeline::build_draft_input_ids(int64_t last_token) const {
    // Build [last_token, mask, mask, ..., mask] of size block_size
    ov::Tensor input_ids(ov::element::i64, {1, m_block_size});
    auto* ptr = input_ids.data<int64_t>();
    ptr[0] = last_token;
    std::fill(ptr + 1, ptr + m_block_size, m_mask_token_id);
    return input_ids;
}

ov::Tensor StatefulDFlashLLMPipeline::build_draft_position_ids(size_t ctx_len) const {
    // Position IDs cover the full range [0 .. ctx_len + block_size - 1]
    // This is required because rotary embeddings in the draft model are applied
    // to the concatenated K sequence (context + query)
    const size_t total_len = ctx_len + m_block_size;
    ov::Tensor position_ids(ov::element::i64, {1, total_len});
    auto* ptr = position_ids.data<int64_t>();
    std::iota(ptr, ptr + total_len, 0);
    return position_ids;
}

void StatefulDFlashLLMPipeline::accumulate_hidden_states(const ov::Tensor& new_hidden, size_t num_accepted) {
    // Slice only the accepted positions from the new hidden states
    // new_hidden shape: [1, verification_window, 5*hidden_size]
    // We want: [1, num_accepted, 5*hidden_size]
    OPENVINO_ASSERT(new_hidden.get_shape()[1] >= num_accepted,
                    "Hidden state has fewer positions than accepted tokens");

    ov::Tensor accepted_hidden = slice_seq(new_hidden, 0, num_accepted);

    if (!m_accumulated_hidden || m_accumulated_hidden.get_size() == 0) {
        // First accumulation — deep copy
        m_accumulated_hidden = ov::Tensor(accepted_hidden.get_element_type(), accepted_hidden.get_shape());
        accepted_hidden.copy_to(m_accumulated_hidden);
    } else {
        // Concatenate along sequence dimension
        m_accumulated_hidden = concat_along_seq(m_accumulated_hidden, accepted_hidden);
    }
}

EncodedResults StatefulDFlashLLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                          const GenerationConfig& config,
                                                          StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulDFlashLLMPipeline::generate(EncodedInputs)");
    generate_timer.start();

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    // Extract input tensors
    ov::Tensor input_ids, attention_mask;
    if (auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }

    OPENVINO_ASSERT(input_ids.get_shape()[0] == 1, "Only batch size 1 supported");
    m_prompt_length = input_ids.get_shape()[1];

    // Reset model states
    m_target->reset_state();
    m_draft->reset_perf_metrics();
    m_accumulated_hidden = ov::Tensor();  // Clear accumulated hidden states

    // Prepare sampling config
    auto sampling_config = config;
    // Inflate max_new_tokens by block_size + 1 to give the sequence group and sampler
    // enough room for speculative tokens. The extra +1 accounts for the "bonus" target
    // token generated after accepting draft tokens in the final iteration.
    sampling_config.max_new_tokens = config.max_new_tokens + m_block_size + 1;

    // Initialize target sequence
    m_target->initialize_sequence(input_ids, sampling_config);

    // ========================================
    // Phase 1: Prefill — process all prompt tokens
    // ========================================
    DFlashInferContext prefill_ctx;
    prefill_ctx.input_token_count = m_prompt_length;
    auto prefill_result = m_target->forward(prefill_ctx);
    OPENVINO_ASSERT(prefill_result.sampled_tokens.size() == 1, "Expected single token from prefill");
    auto initial_token = prefill_result.sampled_tokens[0];

    // Store full prefill hidden states as initial accumulated context
    // Shape: [1, prompt_len, 5*hidden_size]
    auto prefill_hidden = prefill_result.output.hidden_features;
    m_accumulated_hidden = ov::Tensor(prefill_hidden.get_element_type(), prefill_hidden.get_shape());
    prefill_hidden.copy_to(m_accumulated_hidden);

    auto streaming_status = stream_generated_tokens(streamer_ptr, {initial_token});

    // ========================================
    // Phase 2: Speculative Decoding Loop
    // ========================================
    size_t generated_tokens = 1;
    size_t total_draft_accepted = 0;
    size_t total_draft_generated = 0;
    bool eos_reached = false;

    while (!eos_reached && generated_tokens < config.max_new_tokens &&
           m_target->get_sequence_length() < m_prompt_length + config.max_new_tokens &&
           streaming_status == ov::genai::StreamingStatus::RUNNING) {

        auto result = run_speculative_iteration(static_cast<int64_t>(config.eos_token_id),
                                                generated_tokens,
                                                config.max_new_tokens);

        streaming_status = stream_generated_tokens(streamer_ptr, result.validated_tokens);

        // Update statistics
        // DFlash generates (block_size - 1) draft tokens per round
        // (first position is the last accepted token, not a draft)
        total_draft_generated += (m_block_size - 1);
        total_draft_accepted += result.accepted_tokens_count;
        generated_tokens += result.validated_tokens.size();
        eos_reached = result.eos_reached;
    }

    // ========================================
    // Phase 3: Finalization
    // ========================================
    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr)
        streamer_ptr->end();

    // Collect results
    EncodedResults results;
    results.tokens = {m_target->get_generated_tokens()};
    results.scores = {0.0f};

    generate_timer.end();

    // Update performance metrics
    m_sd_perf_metrics.num_input_tokens = m_prompt_length;
    m_sd_perf_metrics.load_time = m_load_time_ms;
    m_sd_perf_metrics.num_accepted_tokens = total_draft_accepted;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    m_sd_perf_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.draft_model_metrics.m_evaluated = false;

    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_target->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft->get_raw_perf_metrics();

    if (total_draft_generated > 0) {
        float acceptance_rate = static_cast<float>(total_draft_accepted) / total_draft_generated * 100.0f;
        m_sd_metrics.update_acceptance_rate(0, acceptance_rate);
        m_sd_metrics.update_draft_accepted_tokens(0, total_draft_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_draft_generated);
        m_sd_metrics.update_generated_len(generated_tokens);
    }

    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    generate_timer.clear();

    return results;
}

StatefulDFlashLLMPipeline::SpeculativeResult StatefulDFlashLLMPipeline::run_speculative_iteration(
    int64_t eos_token_id,
    size_t current_generated_tokens,
    size_t max_new_tokens) {

    SpeculativeResult result;
    const size_t bs = m_block_size;

    OPENVINO_ASSERT(m_target->get_sequence_group(), "Target sequence group not initialized");
    OPENVINO_ASSERT(m_accumulated_hidden && m_accumulated_hidden.get_size() > 0,
                    "Accumulated hidden states must be available for drafting");

    // Get the last accepted token (most recently generated)
    const auto& generated = m_target->get_generated_tokens();
    OPENVINO_ASSERT(!generated.empty(), "No generated tokens available");
    int64_t last_token = generated.back();



    // ========================================
    // Step 1: Draft — single non-causal pass
    // ========================================
    // Build draft inputs:
    //   input_ids   = [last_token, mask, mask, ..., mask]  shape [1, block_size]
    //   hidden      = accumulated target hidden states      shape [1, ctx_len, 5*hidden_size]
    //   position_ids = [0, 1, ..., ctx_len + block_size - 1] shape [1, ctx_len + block_size]

    auto draft_input_ids = build_draft_input_ids(last_token);
    const size_t ctx_len = m_accumulated_hidden.get_shape()[1];
    auto draft_position_ids = build_draft_position_ids(ctx_len);

    // Cast accumulated hidden to f32 if needed (draft model expects f32)
    ov::Tensor draft_hidden = m_accumulated_hidden;

    // Run single non-causal draft inference
    ov::Tensor draft_logits = m_draft->infer(draft_input_ids, draft_hidden, draft_position_ids);

    // Extract draft tokens from positions [1:] (position 0 corresponds to last_token)
    // draft_logits shape: [1, block_size, vocab_size]
    const auto dl_shape = draft_logits.get_shape();
    OPENVINO_ASSERT(dl_shape.size() == 3 && dl_shape[1] == bs,
                    "Draft logits shape mismatch, expected block_size=", bs);

    std::vector<int64_t> draft_tokens;
    draft_tokens.reserve(bs - 1);
    {
        const float* logits_ptr = draft_logits.data<float>();
        const size_t vocab_size = dl_shape[2];
        // Draft sampling is greedy-only (argmax). This is intentional:
        // DFlash draft model outputs are verified against target, so temperature/top-p
        // would only apply to target sampling. If generation config uses non-greedy
        // sampling, draft tokens are still greedily selected for maximum acceptance rate.
        // Skip position 0 (last_token), take argmax from positions [1..block_size-1]
        for (size_t pos = 1; pos < bs; ++pos) {
            const float* pos_logits = logits_ptr + pos * vocab_size;
            auto max_it = std::max_element(pos_logits, pos_logits + vocab_size);
            draft_tokens.push_back(static_cast<int64_t>(std::distance(pos_logits, max_it)));
        }
    }

    // ========================================
    // Step 2: Verify — target model validates draft tokens
    // ========================================
    // Build verification input: [last_token, draft_token_1, ..., draft_token_{bs-1}]
    // The target model processes all bs tokens in one forward pass

    // Append all draft tokens to target sequence speculatively
    m_target->append_tokens(draft_tokens);

    // Verification: target processes the full block
    DFlashInferContext val_ctx;
    val_ctx.input_token_count = bs;           // [last_token, draft_1, ..., draft_{bs-1}]
    val_ctx.sample_count = bs;                // Sample all positions
    val_ctx.num_tokens_to_validate = bs - 1;  // Validate draft_tokens (all except last_token)
    auto val_result = m_target->forward(val_ctx);

    // Sampler validates draft tokens and returns: [accepted_drafts..., new_target_token]
    auto validated_tokens = val_result.sampled_tokens;
    const size_t accepted_count = validated_tokens.size() - 1;  // Minus the new target token
    const int64_t target_predicted_token = validated_tokens.back();
    const size_t tokens_to_remove_from_kv = (bs - 1) - accepted_count;
    size_t total_accepted_positions = validated_tokens.size();  // accepted + 1 (the new token)

    // ========================================
    // Step 3: Check max_new_tokens limit
    // ========================================
    size_t tokens_after_accept = current_generated_tokens + validated_tokens.size();
    if (tokens_after_accept > max_new_tokens) {
        size_t excess_tokens = tokens_after_accept - max_new_tokens;
        OPENVINO_ASSERT(excess_tokens < validated_tokens.size());
        size_t tokens_to_keep = validated_tokens.size() - excess_tokens;

        validated_tokens.resize(tokens_to_keep);
        total_accepted_positions = tokens_to_keep;

        m_target->truncate_sequence(m_prompt_length + max_new_tokens);

        auto& target_batch_sizes = m_target->get_raw_perf_metrics().m_batch_sizes;
        OPENVINO_ASSERT(!target_batch_sizes.empty());
        target_batch_sizes.back() = tokens_to_keep;
    }

    // ========================================
    // Step 4: Synchronize KV cache
    // ========================================
    // Trim rejected KV cache entries from target model
    if (tokens_to_remove_from_kv > 0) {
        m_target->trim_kv_cache(tokens_to_remove_from_kv);
    }

    // ========================================
    // Step 5: Accumulate hidden states for next draft pass
    // ========================================
    // val_result.output.hidden_features: [1, bs, 5*hidden_size]
    // We only keep the first `total_accepted_positions` entries
    auto verify_hidden = val_result.output.hidden_features;
    OPENVINO_ASSERT(verify_hidden && verify_hidden.get_size() > 0, "Missing verification hidden features");
    accumulate_hidden_states(verify_hidden, total_accepted_positions);

    // ========================================
    // Step 6: Build result
    // ========================================
    result.accepted_tokens_count = accepted_count;
    result.validated_tokens = std::move(validated_tokens);
    result.eos_reached = (target_predicted_token == eos_token_id);

    return result;
}

void StatefulDFlashLLMPipeline::finish_chat() {
    StatefulSpeculativePipelineBase::finish_chat();
}

SpeculativeDecodingMetrics StatefulDFlashLLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

// ============================================================================
// DFlash model transform utilities
// ============================================================================
namespace utils {
namespace dflash {

void transform_target_hidden_state(std::shared_ptr<ov::Model>& model,
                                   const std::vector<int32_t>& target_layer_ids) {
    // DFlash extracts hidden states from 5 specific target model layers
    // and concatenates them along the last dimension.
    //
    // This is similar to Eagle3's transform_hidden_state but with 5 layers
    // instead of 3, and no FC movement (FC stays in draft model).
    //
    // For Qwen3-4B with target_layer_ids = [1, 9, 17, 25, 33]:
    //   output "last_hidden_state" = Concat([
    //       residual_add_layer_1,   // [batch, seq, 2560]
    //       residual_add_layer_9,   // [batch, seq, 2560]
    //       residual_add_layer_17,  // [batch, seq, 2560]
    //       residual_add_layer_25,  // [batch, seq, 2560]
    //       residual_add_layer_33,  // [batch, seq, 2560]
    //   ], dim=-1)  → [batch, seq, 12800]

    if (target_layer_ids.empty()) {
        return;
    }

    OPENVINO_ASSERT(target_layer_ids.size() == 5,
                    "DFlash requires exactly 5 target layers for feature extraction, got: ",
                    target_layer_ids.size());

    // Build pattern strings for each target layer
    std::vector<std::string> patterns;
    patterns.reserve(target_layer_ids.size());
    for (int32_t idx : target_layer_ids) {
        patterns.emplace_back("layers." + std::to_string(idx) + "/");
    }

    // Helper: check if node is a residual Add node.
    // NOTE: This heuristic is Qwen3-specific — it assumes the pattern:
    //   Add(x, MatMul(Multiply(...), ...)) which matches the standard HuggingFace
    //   export structure for Qwen3 models. Other model families or optimized graphs
    //   may require a different detection strategy.
    auto is_residual_node = [](const std::shared_ptr<ov::Node>& node) -> bool {
        if (const auto& add = ov::as_type_ptr<ov::op::v1::Add>(node)) {
            auto input1 = add->get_input_node_shared_ptr(1);
            auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(input1);
            if (!matmul) return false;
            auto matmul_input = matmul->get_input_node_shared_ptr(0);
            return matmul_input && ov::is_type<ov::op::v1::Multiply>(matmul_input);
        }
        return false;
    };

    std::vector<ov::Output<ov::Node>> residual_outputs;
    for (const auto& node : model->get_ordered_ops()) {
        if (!is_residual_node(node)) continue;
        const std::string& name = node->get_friendly_name();
        for (const auto& pattern : patterns) {
            if (name.find(pattern) != std::string::npos) {
                residual_outputs.push_back(node->output(0));
                break;
            }
        }
    }

    OPENVINO_ASSERT(residual_outputs.size() == patterns.size(),
                    "Expected ", patterns.size(), " hidden states but found ", residual_outputs.size(),
                    ". Check that target_layer_ids match the target model's layer numbering.");

    // Concatenate all 5 hidden states along last dimension
    auto concat = std::make_shared<ov::op::v0::Concat>(residual_outputs, -1);
    concat->set_friendly_name("dflash_hidden_states_concat");

    // Add as new result output
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    const std::string output_name = "last_hidden_state";
    result->output(0).set_names({output_name});
    result->set_friendly_name(output_name);
    result->get_rt_info()["manually_added_output"] = true;
    model->add_results({result});
}

DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config) {
    DFlashRTInfo rt_info;
    if (config.find("dflash_mode") != config.end()) {
        rt_info.dflash_mode = config.at("dflash_mode").as<bool>();
        config.erase("dflash_mode");

        // Extract block_size
        auto it_bs = config.find("block_size");
        if (it_bs != config.end()) {
            rt_info.block_size = it_bs->second.as<int32_t>();
            config.erase("block_size");
        }

        // Extract mask_token_id
        auto it_mask = config.find("mask_token_id");
        if (it_mask != config.end()) {
            rt_info.mask_token_id = it_mask->second.as<int64_t>();
            config.erase("mask_token_id");
        }

        // Extract target_layer_ids
        auto it_layers = config.find("target_layer_ids");
        if (it_layers != config.end()) {
            OPENVINO_ASSERT(it_layers->second.is<std::vector<int32_t>>(),
                            "target_layer_ids must be a vector of int32_t");
            rt_info.target_layer_ids = it_layers->second.as<std::vector<int32_t>>();
            config.erase("target_layer_ids");
        } else {
            // Default for Qwen3-4B based DFlash
            rt_info.target_layer_ids = {1, 9, 17, 25, 33};
        }

        OPENVINO_ASSERT(rt_info.target_layer_ids.size() == 5,
                        "DFlash requires exactly 5 target layers, got: ", rt_info.target_layer_ids.size());
    }
    return rt_info;
}

void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (model->has_rt_info("dflash_mode") && model->get_rt_info<bool>("dflash_mode")) {
        properties["dflash_mode"] = true;
        if (model->has_rt_info("block_size")) {
            properties["block_size"] = model->get_rt_info<int>("block_size");
        }
        if (model->has_rt_info("mask_token_id")) {
            properties["mask_token_id"] = model->get_rt_info<int64_t>("mask_token_id");
        }
        if (model->has_rt_info("target_layer_ids")) {
            properties["target_layer_ids"] = model->get_rt_info<std::vector<int32_t>>("target_layer_ids");
        }
    }
}

}  // namespace dflash
}  // namespace utils

}  // namespace ov::genai
