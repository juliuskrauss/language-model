#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <torch/torch.h>
#include <memory>

class SelfAttentionHead : public torch::nn::Module {
public:
    SelfAttentionHead(int context_len, int embedding_dim, int parameter_dim);
    torch::Tensor forward(torch::Tensor x);

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;

    torch::nn::Linear Q{nullptr};
    torch::nn::Linear K{nullptr};
    torch::nn::Linear V{nullptr};
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

#endif // TRANSFORMER_H
