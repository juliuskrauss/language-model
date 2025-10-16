#include <torch/torch.h>

class Transformer {
public: 
    Transformer () {

    }

    void train() {}

    torch::Tensor predict() {}
    
private:
};

class AttentionHead {
    public:
        AttentionHead (int dim) {
            Q = torch::rand({dim, dim});
            K = torch::rand({dim, dim});
            V = torch::rand({dim, dim});
        }
        void attention () {}

    private:
    torch::Tensor Q;
    torch::Tensor K;
    torch::Tensor V;
};

