#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <torch/torch.h>
#include <memory>

class SelfAttentionHead : public torch::nn::Module {
public:
    SelfAttentionHead(int context_len, int embedding_dim, int parameter_dim, bool is_causal = false);
    torch::Tensor forward(torch::Tensor x);

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;
    bool is_causal;

    torch::nn::Linear Q{nullptr};
    torch::nn::Linear K{nullptr};
    torch::nn::Linear V{nullptr};
    torch::Tensor causal_mask;
};

class MultiHeadAttention : public torch::nn::Module {
public:
    MultiHeadAttention(int context_len, int embedding_dim, int num_heads, bool is_causal = false, float dropout_p = 0.1);
    torch::Tensor forward(torch::Tensor x);

private:
    int context_len;
    int embedding_dim;
    int num_heads;
    int head_dim;
    bool is_causal;
    
    std::vector<std::shared_ptr<SelfAttentionHead>> heads;
    torch::nn::Linear output_projection{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::Tensor causal_mask;
};

class FeedForward : public torch::nn::Module {
public:
    FeedForward(int embedding_dim, int hidden_dim, float dropout_p = 0.1);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Dropout dropout{nullptr};
};

class TransformerBlock : public torch::nn::Module {
public:
    TransformerBlock(int context_len, int embedding_dim, int num_heads, float dropout_p = 0.1);
    torch::Tensor forward(torch::Tensor x);

private:
    int embedding_dim;
    
    std::shared_ptr<MultiHeadAttention> attention;
    std::shared_ptr<FeedForward> feed_forward;
    torch::nn::LayerNorm ln1{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    torch::nn::Dropout dropout{nullptr};
};

class Encoder : public torch::nn::Module {
public:
    Encoder(int context_len, int embedding_dim, int parameter_dim);
    torch::Tensor forward(torch::Tensor x);

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;
    
    std::shared_ptr<SelfAttentionHead> self_attention_head;
    torch::nn::Linear feed_forward{nullptr};
};

class EncoderDecoderAttentionHead : public torch::nn::Module {
public:
    EncoderDecoderAttentionHead(int context_len, int embedding_dim, int parameter_dim);
    torch::Tensor forward(torch::Tensor decoder_input, torch::Tensor encoder_output);

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;

    torch::nn::Linear Q{nullptr};
    torch::nn::Linear K{nullptr};
    torch::nn::Linear V{nullptr};
};

class Decoder : public torch::nn::Module {
public:
    Decoder(int context_len, int embedding_dim, int parameter_dim);
    torch::Tensor forward(torch::Tensor x, torch::Tensor encoder_output);

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;
    
    std::shared_ptr<SelfAttentionHead> self_attention_head;
    std::shared_ptr<EncoderDecoderAttentionHead> cross_attention_head;
    torch::nn::Linear feed_forward{nullptr};
};

class Transformer : public torch::nn::Module {
public:
    Transformer(int context_len, int embedding_dim, int parameter_dim);
    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt);

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;
    
    std::shared_ptr<Encoder> encoder;
    std::shared_ptr<Decoder> decoder;
};

// Decoder-only transformer for language modeling
class DecoderOnlyTransformer : public torch::nn::Module {
public:
    DecoderOnlyTransformer(int context_len, int embedding_dim, int num_heads, int num_layers, float dropout_p = 0.1);
    torch::Tensor forward(torch::Tensor x);

private:
    int context_len;
    int embedding_dim;
    
    std::vector<std::shared_ptr<TransformerBlock>> layers;
    torch::nn::LayerNorm final_ln{nullptr};
};

#endif // TRANSFORMER_H
