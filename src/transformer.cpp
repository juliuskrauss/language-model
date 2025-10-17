#include <torch/torch.h>
#include <cmath>
#include <vector>

class Transformer {
public: 
    Transformer () {

    }

    void train() {}

    torch::Tensor predict() {}
    
private:
};

class Encoder : public torch::nn::Module {
public:
    Encoder(int context_len, int embedding_dim, int parameter_dim)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim),
          attention_head(context_len, embedding_dim, parameter_dim),
          L1(torch::nn::Linear(context_len, embedding_dim))
          {
            register_module("L1", L1);
          }

    torch::Tensor encode(torch::Tensor x) {
        torch::Tensor mask = attention_head.attention(x);

        x += mask;
        std::vector<int64_t> x_dim = {x.size(1)};
        x = torch::nn::functional::layer_norm(x, x_dim); 

        auto L1_out = L1->forward(x);

        x += L1_out;
        return torch::nn::functional::layer_norm(x, x_dim);
    }

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;
    torch::nn::Linear L1{nullptr};
    AttentionHead attention_head;  
};

class Decoder : public torch::nn::Module {
public:

private:

};

class AttentionHead : public torch::nn::Module {
public:
    AttentionHead(int context_len, int embedding_dim, int parameter_dim)
        : context_len(context_len),
          embedding_dim(embedding_dim),
          parameter_dim(parameter_dim),
          Q(torch::nn::Linear(embedding_dim, parameter_dim)),
          K(torch::nn::Linear(embedding_dim, parameter_dim)),
          V(torch::nn::Linear(embedding_dim, embedding_dim))
    {
        // Register submodules so their parameters are tracked
        register_module("Q", Q);
        register_module("K", K);
        register_module("V", V);
    }

    torch::Tensor attention(torch::Tensor x) {
        // x: input tensor of shape (seq_len, embedding_dim)
        auto Q_out = Q->forward(x);  // shape (seq_len, parameter_dim)
        auto K_out = K->forward(x);  // shape (seq_len, parameter_dim)
        auto V_out = V->forward(x);  // shape (seq_len, embedding_dim)

        auto cross_product = torch::matmul(Q_out, K_out.t());
        auto normalized = torch::nn::functional::softmax(
            cross_product / std::sqrt(static_cast<double>(parameter_dim)),
            torch::nn::functional::SoftmaxFuncOptions(1));

        return torch::matmul(normalized, V_out);
    }

private:
    int context_len;
    int embedding_dim;
    int parameter_dim;

    torch::nn::Linear Q{nullptr};
    torch::nn::Linear K{nullptr};
    torch::nn::Linear V{nullptr};
};


