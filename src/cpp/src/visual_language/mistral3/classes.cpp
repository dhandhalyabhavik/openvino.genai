// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/mistral3/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

/**
 * @brief Pixtral-style image preprocessing for Mistral3.
 *
 * Resize so that the longest edge equals `longest_edge` (from preprocessor_config.json),
 * both dimensions are made divisible by patch_size, then normalize with CLIP mean/std.
 */
clip_image_f32 preprocess_clip_image_mistral3(const clip_image_u8& image, const ProcessorConfig& config) {
    int longest_edge = static_cast<int>(config.size_longest_edge);
    int patch_size = static_cast<int>(config.patch_size);

    // Compute scale so longest edge = longest_edge
    int max_dim = std::max(image.nx, image.ny);
    float scale = static_cast<float>(longest_edge) / static_cast<float>(max_dim);

    int new_width = static_cast<int>(image.nx * scale);
    int new_height = static_cast<int>(image.ny * scale);

    // Make dimensions divisible by patch_size
    new_width = (new_width / patch_size) * patch_size;
    new_height = (new_height / patch_size) * patch_size;

    // Ensure at least one patch in each dimension
    new_width = std::max(new_width, patch_size);
    new_height = std::max(new_height, patch_size);

    // Resize
    clip_image_u8 resized_image;
    bilinear_resize(image, resized_image, new_width, new_height);

    // Normalize with CLIP mean/std
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);
    return normalized_image;
}

ov::Tensor get_pixel_values_mistral3(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_mistral3(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
}

} // namespace


EncodedImage VisionEncoderMistral3::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values = get_pixel_values_mistral3(image, config);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    return {std::move(image_features)};
}


InputsEmbedderMistral3::InputsEmbedderMistral3(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) { }

InputsEmbedderMistral3::InputsEmbedderMistral3(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

std::vector<ov::genai::EncodedImage> InputsEmbedderMistral3::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderMistral3::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images
) const {
    // Mistral3 uses [IMG] token (token id = image_token_index) as the image placeholder.
    // The image token string representation varies, but in the tokenizer it maps to token id 10.
    // For text normalization we use a placeholder string that maps to the image token.
    std::string image_token = "[IMG]";

    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    // Expand each image_token occurrence to the correct number of tokens
    // based on the vision embeddings output size.
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const auto& encoded = images.at(new_image_id - base_id);
        // The vision embeddings output shape is [num_tokens, hidden_size]
        size_t num_image_tokens = encoded.resized_source.get_shape().at(0);

        std::string expanded_tag;
        for (size_t i = 0; i < num_image_tokens; ++i) {
            expanded_tag += image_token;
        }

        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, image_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderMistral3::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence
) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
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

    // Use the image_token_index from config (default 10) to identify image placeholder tokens
    int64_t image_token_id = m_vlm_config.image_token_index;
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

} // namespace ov::genai
