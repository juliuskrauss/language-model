#include <torch/torch.h>
#include <cmath>
#include <vector>

class SelfAttentionHead : public torch::nn::Module {
public:
    SelfAttentionHead(int context_len, int embedding_dim, int parameter_dim)
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

    torch::Tensor forward(torch::Tensor x) {
        auto Q_out = Q->forward(x);
        auto K_out = K->forward(x);
        auto V_out = V->forward(x);

        auto scores = torch::matmul(Q_out, K_out.transpose(-2, -1)) / std::sqrt(static_cast<double>(parameter_dim));
        auto attention_weights = torch::nn::functional::softmax(scores, torch::nn::functional::SoftmaxFuncOptions(-1));

        return torch::matmul(attention_weights, V_out);
    }

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
    Encoder(int context_len, int embedding_dim, int parameter_dim)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim)
    {
        self_attention_head = register_module("self_attention", std::make_shared<SelfAttentionHead>(context_len, embedding_dim, parameter_dim));
        feed_forward = register_module("feed_forward", torch::nn::Linear(embedding_dim, embedding_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto attention_out = self_attention_head->forward(x);
        x = x + attention_out;
        x = torch::nn::functional::layer_norm(x, torch::nn::functional::LayerNormFuncOptions({embedding_dim}));

        auto ff_out = feed_forward->forward(x);
        x = x + ff_out;
        x = torch::nn::functional::layer_norm(x, torch::nn::functional::LayerNormFuncOptions({embedding_dim}));

        return x;
    }

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;
    
    std::shared_ptr<SelfAttentionHead> self_attention_head;
    torch::nn::Linear feed_forward{nullptr};
};

class EncoderDecoderAttentionHead : public torch::nn::Module {
public:
    EncoderDecoderAttentionHead(int context_len, int embedding_dim, int parameter_dim)
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

    torch::Tensor forward(torch::Tensor decoder_input, torch::Tensor encoder_output) {
        auto Q_out = Q->forward(decoder_input);
        auto K_out = K->forward(encoder_output);
        auto V_out = V->forward(encoder_output);

        auto scores = torch::matmul(Q_out, K_out.transpose(-2, -1)) / std::sqrt(static_cast<double>(parameter_dim));
        auto attention_weights = torch::nn::functional::softmax(scores, torch::nn::functional::SoftmaxFuncOptions(-1));

        return torch::matmul(attention_weights, V_out);
    }

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
    Decoder(int context_len, int embedding_dim, int parameter_dim)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim)
    {
        self_attention_head = register_module("self_attention", std::make_shared<SelfAttentionHead>(context_len, embedding_dim, parameter_dim));
        cross_attention_head = register_module("cross_attention", std::make_shared<EncoderDecoderAttentionHead>(context_len, embedding_dim, parameter_dim));
        feed_forward = register_module("feed_forward", torch::nn::Linear(embedding_dim, embedding_dim));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor encoder_output) {
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
    Transformer(int context_len, int embedding_dim, int parameter_dim)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim)
    {
        encoder = register_module("encoder", std::make_shared<Encoder>(context_len, embedding_dim, parameter_dim));
        decoder = register_module("decoder", std::make_shared<Decoder>(context_len, embedding_dim, parameter_dim));
    }

    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt) {
        auto encoder_output = encoder->forward(src);
        auto decoder_output = decoder->forward(tgt, encoder_output);
        return decoder_output;
    }

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;
    
    std::shared_ptr<Encoder> encoder;
    std::shared_ptr<Decoder> decoder;
};
