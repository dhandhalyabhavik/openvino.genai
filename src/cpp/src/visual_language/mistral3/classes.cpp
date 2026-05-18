// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/mistral3/classes.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "visual_language/clip.hpp"
#include "utils.hpp"

namespace ov::genai {
namespace {

ImageSize get_pixtral_resize_output_size(size_t height, size_t width, size_t longest_edge, size_t patch_size) {
    double ratio = std::max(static_cast<double>(height) / static_cast<double>(longest_edge),
                            static_cast<double>(width) / static_cast<double>(longest_edge));
    if (ratio > 1.0) {
        height = static_cast<size_t>(std::floor(static_cast<double>(height) / ratio));
        width = static_cast<size_t>(std::floor(static_cast<double>(width) / ratio));
    }

    const size_t num_height_tokens = (height - 1) / patch_size + 1;
    const size_t num_width_tokens = (width - 1) / patch_size + 1;
    return {num_height_tokens * patch_size, num_width_tokens * patch_size};
}

ov::Tensor get_pixel_values_pixtral(const ov::Tensor& image, const ProcessorConfig& config, size_t spatial_merge_size, ImageSize& resized_size) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    const size_t prompt_patch_size = config.patch_size * spatial_merge_size;
    resized_size = get_pixtral_resize_output_size(input_image.ny, input_image.nx, config.size_longest_edge, prompt_patch_size);

    clip_image_u8 resized_image;
    if (static_cast<size_t>(input_image.ny) != resized_size.height || static_cast<size_t>(input_image.nx) != resized_size.width) {
        bicubic_resize(input_image, resized_image, static_cast<int>(resized_size.width), static_cast<int>(resized_size.height));
    } else {
        resized_image = std::move(input_image);
    }

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);
    return clip_image_f32_to_tensor(normalized_image);
}

} // namespace

VisionEncoderMistral3::VisionEncoderMistral3(const std::filesystem::path& model_dir,
                                             const std::string& device,
                                             const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties) {
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
    m_spatial_merge_size = vlm_config.spatial_merge_size;
}

VisionEncoderMistral3::VisionEncoderMistral3(const ModelsMap& models_map,
                                             const std::filesystem::path& config_dir_path,
                                             const std::string& device,
                                             const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, device, properties) {
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    m_spatial_merge_size = vlm_config.spatial_merge_size;
}

EncodedImage VisionEncoderMistral3::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    ImageSize resized_pixel_size;
    ov::Tensor pixel_values = get_pixel_values_pixtral(image, config, m_spatial_merge_size, resized_pixel_size);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    OPENVINO_ASSERT(infer_output.get_shape().size() == 2,
        "Mistral3 vision embeddings output is expected to have rank 2 [tokens, hidden_size], got ", infer_output.get_shape());

    ov::Shape image_features_shape = infer_output.get_shape();
    ov::Shape batched_shape{1, image_features_shape.at(0), image_features_shape.at(1)};
    ov::Tensor image_features(infer_output.get_element_type(), batched_shape);
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    const size_t prompt_patch_size = config.patch_size * m_spatial_merge_size;
    ImageSize prompt_grid_size{resized_pixel_size.height / prompt_patch_size,
                               resized_pixel_size.width / prompt_patch_size};

    return {std::move(image_features), prompt_grid_size};
}

InputsEmbedderMistral3::InputsEmbedderMistral3(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    // PixtralProcessor's reference path tokenizes the rendered chat template with
    // tokenizer defaults, which adds a BOS token even though the template already
    // starts with <s>. Preserve that behavior for parity with Transformers.
    m_add_special_tokens = true;
    m_add_special_tokens_is_set = true;
}

InputsEmbedderMistral3::InputsEmbedderMistral3(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    m_add_special_tokens = true;
    m_add_special_tokens_is_set = true;
}

NormalizedPrompt InputsEmbedderMistral3::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, m_vlm_config.mistral3_image_token, m_vlm_config.mistral3_image_token, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t image_id : images_sequence) {
        const auto& encoded_image = images.at(image_id - base_id);
        const size_t grid_h = encoded_image.resized_source_size.height;
        const size_t grid_w = encoded_image.resized_source_size.width;

        std::string expanded_tag;
        expanded_tag.reserve((m_vlm_config.mistral3_image_token.size() * grid_w + m_vlm_config.mistral3_image_break_token.size()) * grid_h);
        for (size_t h = 0; h < grid_h; ++h) {
            for (size_t w = 0; w < grid_w; ++w) {
                expanded_tag += m_vlm_config.mistral3_image_token;
            }
            expanded_tag += (h + 1 == grid_h) ? m_vlm_config.mistral3_image_end_token : m_vlm_config.mistral3_image_break_token;
        }

        searched_pos = unified_prompt.find(m_vlm_config.mistral3_image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos, "Mistral3 image token was not found in normalized prompt");
        unified_prompt.replace(searched_pos, m_vlm_config.mistral3_image_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderMistral3::get_inputs_embeds(const std::string& unified_prompt,
                                                     const std::vector<ov::genai::EncodedImage>& images,
                                                     ov::genai::VLMPerfMetrics& metrics,
                                                     bool recalculate_merged_embeddings,
                                                     const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t image_id : images_sequence) {
        image_embeds.push_back(images.at(image_id).resized_source);
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    const int64_t image_token_id = m_vlm_config.mistral3_image_token_index;

    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    const auto input_ids_shape = input_ids.get_shape();
    const auto embeds_shape = inputs_embeds.get_shape();
    OPENVINO_ASSERT(input_ids_shape.size() == 2 && embeds_shape.size() == 3,
        "Unexpected Mistral3 input/embedding ranks: ", input_ids_shape, " and ", embeds_shape);
    const size_t batch_size = input_ids_shape.at(0);
    const size_t seq_len = input_ids_shape.at(1);
    const size_t hidden_size = embeds_shape.at(2);
    OPENVINO_ASSERT(batch_size == 1, "Only batch size 1 is supported for Mistral3 VLM inputs");

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    float* inputs_embeds_data = inputs_embeds.data<float>();

    size_t image_idx = 0;
    size_t token_idx_in_image = 0;
    for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
        if (input_ids_data[seq_idx] != image_token_id) {
            continue;
        }
        OPENVINO_ASSERT(image_idx < image_embeds.size(), "More Mistral3 image tokens than provided image embeddings");
        const auto& current_image = image_embeds.at(image_idx);
        const auto image_shape = current_image.get_shape();
        OPENVINO_ASSERT(image_shape.size() == 3 && image_shape.at(0) == 1 && image_shape.at(2) == hidden_size,
            "Unexpected Mistral3 image embedding shape ", image_shape, ", hidden size ", hidden_size);

        if (token_idx_in_image >= image_shape.at(1)) {
            ++image_idx;
            token_idx_in_image = 0;
            OPENVINO_ASSERT(image_idx < image_embeds.size(), "More Mistral3 image tokens than provided image embeddings");
        }

        const float* image_data = current_image.data<const float>() + token_idx_in_image * hidden_size;
        float* dst = inputs_embeds_data + seq_idx * hidden_size;
        std::copy_n(image_data, hidden_size, dst);
        ++token_idx_in_image;
    }

    if (!image_embeds.empty()) {
        OPENVINO_ASSERT(image_idx == image_embeds.size() - 1 && token_idx_in_image == image_embeds.back().get_shape().at(1),
            "Number of Mistral3 image embeddings does not match image tokens in prompt");
    }

    return inputs_embeds;
}

} // namespace ov::genai
