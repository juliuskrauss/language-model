#include "../include/transformer.h"
#include <cmath>

SelfAttentionHead::SelfAttentionHead(int context_len, int embedding_dim, int parameter_dim, bool is_causal)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim),
          is_causal(is_causal),
          Q(torch::nn::Linear(embedding_dim, parameter_dim)),
          K(torch::nn::Linear(embedding_dim, parameter_dim)),
          V(torch::nn::Linear(embedding_dim, parameter_dim))
{
    register_module("Q", Q);
    register_module("K", K);
    register_module("V", V);
    
    if (is_causal) {
        causal_mask = torch::triu(torch::ones({context_len, context_len}), 1).to(torch::kBool);
    }
}

torch::Tensor SelfAttentionHead::forward(torch::Tensor x) {
    auto Q_out = Q->forward(x);
    auto K_out = K->forward(x);
    auto V_out = V->forward(x);

    auto scores = torch::matmul(Q_out, K_out.transpose(-2, -1)) / std::sqrt(static_cast<double>(parameter_dim));
    
    if (is_causal) {
        auto seq_len = scores.size(-1);
        auto mask = causal_mask.slice(0, 0, seq_len).slice(1, 0, seq_len);
        scores = scores.masked_fill(mask.to(scores.device()), -1e9);
    }
    
    auto attention_weights = torch::nn::functional::softmax(scores, torch::nn::functional::SoftmaxFuncOptions(-1));

    return torch::matmul(attention_weights, V_out);
}

MultiHeadAttention::MultiHeadAttention(int context_len, int embedding_dim, int num_heads, bool is_causal, float dropout_p)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          num_heads(num_heads),
          head_dim(embedding_dim / num_heads),
          is_causal(is_causal),
          output_projection(torch::nn::Linear(embedding_dim, embedding_dim)),
          dropout(torch::nn::Dropout(dropout_p))
{
    for (int i = 0; i < num_heads; ++i) {
        auto head = register_module("head_" + std::to_string(i), 
            std::make_shared<SelfAttentionHead>(context_len, embedding_dim, head_dim, is_causal));
        heads.push_back(head);
    }
    
    register_module("output_projection", output_projection);
    
    if (is_causal) {
        causal_mask = torch::triu(torch::ones({context_len, context_len}), 1).to(torch::kBool);
    }
}

torch::Tensor MultiHeadAttention::forward(torch::Tensor x) {
    std::vector<torch::Tensor> head_outputs;
    
    for (auto& head : heads) {
        head_outputs.push_back(head->forward(x));
    }
    
    // Concatenate all heads
    auto concatenated = torch::cat(head_outputs, -1);
    
    // Project back to embedding dimension
    auto output = output_projection->forward(concatenated);
    output = dropout->forward(output);
    
    return output;
}

FeedForward::FeedForward(int embedding_dim, int hidden_dim, float dropout_p)
        : fc1(torch::nn::Linear(embedding_dim, hidden_dim)),
          fc2(torch::nn::Linear(hidden_dim, embedding_dim)),
          dropout(torch::nn::Dropout(dropout_p))
{
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor FeedForward::forward(torch::Tensor x) {
    x = fc1->forward(x);
    x = torch::gelu(x);
    x = dropout->forward(x);
    x = fc2->forward(x);
    x = dropout->forward(x);
    return x;
}

TransformerBlock::TransformerBlock(int context_len, int embedding_dim, int num_heads, float dropout_p)
        : embedding_dim(embedding_dim),
          ln1(torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim}))),
          ln2(torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim}))),
          dropout(torch::nn::Dropout(dropout_p))
{
    attention = register_module("attention", 
        std::make_shared<MultiHeadAttention>(context_len, embedding_dim, num_heads, true, dropout_p));
    
    int ff_hidden_dim = 4 * embedding_dim;  // Standard practice: 4x expansion
    feed_forward = register_module("feed_forward", 
        std::make_shared<FeedForward>(embedding_dim, ff_hidden_dim, dropout_p));
    
    register_module("ln1", ln1);
    register_module("ln2", ln2);
}

torch::Tensor TransformerBlock::forward(torch::Tensor x) {
    auto attention_out = attention->forward(x);
    x = x + dropout->forward(attention_out);
    x = ln1->forward(x);

    auto ff_out = feed_forward->forward(x);
    x = x + dropout->forward(ff_out);
    x = ln2->forward(x);
    
    return x;
}

DecoderOnlyTransformer::DecoderOnlyTransformer(int context_len, int embedding_dim, int num_heads, int num_layers, float dropout_p)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          final_ln(torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dim})))
{
    for (int i = 0; i < num_layers; ++i) {
        auto layer = register_module("layer_" + std::to_string(i), 
            std::make_shared<TransformerBlock>(context_len, embedding_dim, num_heads, dropout_p));
        layers.push_back(layer);
    }
    
    register_module("final_ln", final_ln);
}

torch::Tensor DecoderOnlyTransformer::forward(torch::Tensor x) {
    for (auto& layer : layers) {
        x = layer->forward(x);
    }
    
    x = final_ln->forward(x);
    
    return x;
}

Encoder::Encoder(int context_len, int embedding_dim, int parameter_dim)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim)
{
    self_attention_head = register_module("self_attention", std::make_shared<SelfAttentionHead>(context_len, embedding_dim, parameter_dim, false));
    feed_forward = register_module("feed_forward", torch::nn::Linear(embedding_dim, embedding_dim));
}

torch::Tensor Encoder::forward(torch::Tensor x) {
    auto attention_out = self_attention_head->forward(x);
    x = x + attention_out;
    x = torch::nn::functional::layer_norm(x, torch::nn::functional::LayerNormFuncOptions({embedding_dim}));

    auto ff_out = feed_forward->forward(x);
    x = x + ff_out;
    x = torch::nn::functional::layer_norm(x, torch::nn::functional::LayerNormFuncOptions({embedding_dim}));

    return x;
}

EncoderDecoderAttentionHead::EncoderDecoderAttentionHead(int context_len, int embedding_dim, int parameter_dim)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim),
          Q(torch::nn::Linear(embedding_dim, parameter_dim)),
          K(torch::nn::Linear(embedding_dim, parameter_dim)),
          V(torch::nn::Linear(embedding_dim, embedding_dim))
{
    register_module("Q", Q);
    register_module("K", K);
    register_module("V", V);
}

torch::Tensor EncoderDecoderAttentionHead::forward(torch::Tensor decoder_input, torch::Tensor encoder_output) {
    auto Q_out = Q->forward(decoder_input);
    auto K_out = K->forward(encoder_output);
    auto V_out = V->forward(encoder_output);

    auto scores = torch::matmul(Q_out, K_out.transpose(-2, -1)) / std::sqrt(static_cast<double>(parameter_dim));
    auto attention_weights = torch::nn::functional::softmax(scores, torch::nn::functional::SoftmaxFuncOptions(-1));

    return torch::matmul(attention_weights, V_out);
}

Decoder::Decoder(int context_len, int embedding_dim, int parameter_dim)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim)
{
    self_attention_head = register_module("self_attention", std::make_shared<SelfAttentionHead>(context_len, embedding_dim, parameter_dim, true));
    cross_attention_head = register_module("cross_attention", std::make_shared<EncoderDecoderAttentionHead>(context_len, embedding_dim, parameter_dim));
    feed_forward = register_module("feed_forward", torch::nn::Linear(embedding_dim, embedding_dim));
}

torch::Tensor Decoder::forward(torch::Tensor x, torch::Tensor encoder_output) {
    auto self_attention_out = self_attention_head->forward(x);
    x = x + self_attention_out;
    x = torch::nn::functional::layer_norm(x, torch::nn::functional::LayerNormFuncOptions({embedding_dim}));

    auto cross_attention_out = cross_attention_head->forward(x, encoder_output);
    x = x + cross_attention_out;
    x = torch::nn::functional::layer_norm(x, torch::nn::functional::LayerNormFuncOptions({embedding_dim}));

    auto ff_out = feed_forward->forward(x);
    x = x + ff_out;
    x = torch::nn::functional::layer_norm(x, torch::nn::functional::LayerNormFuncOptions({embedding_dim}));

    return x;
}

Transformer::Transformer(int context_len, int embedding_dim, int parameter_dim)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim)
{
    encoder = register_module("encoder", std::make_shared<Encoder>(context_len, embedding_dim, parameter_dim));
    decoder = register_module("decoder", std::make_shared<Decoder>(context_len, embedding_dim, parameter_dim));
}

torch::Tensor Transformer::forward(torch::Tensor src, torch::Tensor tgt) {
    auto encoder_output = encoder->forward(src);
    auto decoder_output = decoder->forward(tgt, encoder_output);
    return decoder_output;
}
