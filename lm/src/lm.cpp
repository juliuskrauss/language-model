#include "../../transformer/include/transformer.h"
#include <torch/torch.h>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <filesystem>

class CharTokenizer {
public:
    CharTokenizer() {
        // Range of printable ASCII characters 
        for (int i = 32; i < 127; ++i) {
            std::string s(1, static_cast<char>(i));
            char_to_idx[s] = vocab.size();
            vocab.push_back(s);
        }
        
        vocab_size = vocab.size();
    }
    
    std::vector<int> encode(const std::string& text) {
        std::vector<int> tokens;
        for (char c : text) {
            std::string s(1, c);
            if (char_to_idx.count(s)) {
                tokens.push_back(char_to_idx[s]);
            }
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) {
        std::string text;
        for (int idx : tokens) {
            if (idx >= 0 && idx < vocab.size()) {
                text += vocab[idx];
            }
        }
        return text;
    }
    
    int get_vocab_size() const { return vocab_size; }
    
private:
    std::unordered_map<std::string, int> char_to_idx;
    std::vector<std::string> vocab;
    int vocab_size;
};


class TokenEmbedding : public torch::nn::Module {
public:
    TokenEmbedding(int vocab_size, int embedding_dim, int max_seq_len)
        : vocab_size(vocab_size),
          embedding_dim(embedding_dim),
          max_seq_len(max_seq_len),
          E(torch::nn::Embedding(vocab_size, embedding_dim))
    {
        register_module("E", E);
        positional_encoding = create_positional_encoding(max_seq_len, embedding_dim);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto embedding = E->forward(x);
        auto seq_len = x.size(1);
        auto pos_enc = positional_encoding.slice(0, 0, seq_len).unsqueeze(0);
        return embedding + pos_enc;
    }

private:
    int vocab_size;
    int embedding_dim;
    int max_seq_len;
    
    torch::nn::Embedding E{nullptr};
    torch::Tensor positional_encoding;
    
    torch::Tensor create_positional_encoding(int max_len, int d_model) {
        auto pe = torch::zeros({max_len, d_model});
        auto position = torch::arange(0, max_len, torch::kFloat32).unsqueeze(1);
        
        auto div_term = torch::exp(
            torch::arange(0, d_model, 2, torch::kFloat32) * 
            (-std::log(10000.0) / d_model)
        );
        
        pe.index_put_({"...", torch::indexing::Slice(0, torch::indexing::None, 2)}, 
                      torch::sin(position * div_term));
        
        pe.index_put_({"...", torch::indexing::Slice(1, torch::indexing::None, 2)}, 
                      torch::cos(position * div_term));
        
        return pe;
    }
};


class LanguageModel : public torch::nn::Module {
public:
    LanguageModel(int vocab_size, int context_len, int embedding_dim, int num_heads, int num_layers, float dropout = 0.1)
        : vocab_size(vocab_size),
          context_len(context_len),
          embedding_dim(embedding_dim),
          output_projection(torch::nn::Linear(embedding_dim, vocab_size))
    {
        token_embedding = register_module("token_embedding", 
            std::make_shared<TokenEmbedding>(vocab_size, embedding_dim, context_len));
        
        // Use decoder-only transformer for language modeling
        transformer = register_module("transformer", 
            std::make_shared<DecoderOnlyTransformer>(context_len, embedding_dim, num_heads, num_layers, dropout));
        
        register_module("output_projection", output_projection);
    }
    
    torch::Tensor forward(torch::Tensor input_ids) {
        auto embedded = token_embedding->forward(input_ids);
        auto transformer_out = transformer->forward(embedded);
        auto logits = output_projection->forward(transformer_out);
        return logits;
    }
    
    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens, 
                              float temperature, CharTokenizer& tokenizer, bool verbose = false) {
        std::vector<int> tokens = prompt;
        
        this->eval();
        torch::NoGradGuard no_grad;
        
        if (verbose) {
            std::cout << "\n=== Generation Process ===" << std::endl;
            std::cout << "Initial prompt: \"" << tokenizer.decode(prompt) << "\"" << std::endl;
            std::cout << "Generating " << max_new_tokens << " tokens...\n" << std::endl;
        }
        
        for (int i = 0; i < max_new_tokens; ++i) {
            // Use only the last context_len tokens
            int start_idx = std::max(0, static_cast<int>(tokens.size()) - context_len);
            std::vector<int> context(tokens.begin() + start_idx, tokens.end());
            
            auto input = torch::tensor(context, torch::kLong).unsqueeze(0);
            auto logits = this->forward(input);
            
            // Get logits for the last token
            auto next_token_logits = logits.index({0, -1, "..."}) / temperature;
            auto probs = torch::softmax(next_token_logits, -1);
            
            // Sample from the distribution
            auto next_token = torch::multinomial(probs, 1).item<int>();
            
            tokens.push_back(next_token);
            
            if (verbose && (i < 50 || i % 50 == 0 || i == max_new_tokens - 1)) {
                std::string decoded_token = tokenizer.decode({next_token});
                std::cout << "Step " << std::setw(4) << (i + 1) << "/" << max_new_tokens 
                         << ": Token ID: " << std::setw(3) << next_token 
                         << " | Char: '" << decoded_token << "'"
                         << std::endl;
            }
        }
        
        if (verbose) {
            std::cout << "\nGeneration complete!" << std::endl;
        }
        
        return tokens;
    }
    
private:
    int vocab_size;
    int context_len;
    int embedding_dim;
    
    std::shared_ptr<TokenEmbedding> token_embedding;
    std::shared_ptr<DecoderOnlyTransformer> transformer;
    torch::nn::Linear output_projection{nullptr};
};


class DataLoader {
public:
    DataLoader(const std::string& file_path, int batch_size, int seq_len, CharTokenizer& tokenizer)
        : batch_size(batch_size), seq_len(seq_len), tokenizer(tokenizer), current_pos(0)
    {
        load_and_tokenize(file_path);
    }
    
    bool has_next() const {
        return current_pos + batch_size * (seq_len + 1) <= tokens.size();
    }
    
    std::pair<torch::Tensor, torch::Tensor> next_batch() {
        auto batch_tokens = std::vector<std::vector<int>>(batch_size);
        auto batch_targets = std::vector<std::vector<int>>(batch_size);
        
        for (int i = 0; i < batch_size; ++i) {
            int start = current_pos + i * (seq_len + 1);
            batch_tokens[i] = std::vector<int>(tokens.begin() + start, tokens.begin() + start + seq_len);
            batch_targets[i] = std::vector<int>(tokens.begin() + start + 1, tokens.begin() + start + seq_len + 1);
        }
        
        current_pos += batch_size * (seq_len + 1);
        
        auto input_tensor = torch::zeros({batch_size, seq_len}, torch::kLong);
        auto target_tensor = torch::zeros({batch_size, seq_len}, torch::kLong);
        
        for (int i = 0; i < batch_size; ++i) {
            input_tensor[i] = torch::tensor(batch_tokens[i], torch::kLong);
            target_tensor[i] = torch::tensor(batch_targets[i], torch::kLong);
        }
        
        return {input_tensor, target_tensor};
    }
    
    void reset() {
        current_pos = 0;
    }
    
    size_t get_token_count() const {
        return tokens.size();
    }
    
private:
    int batch_size;
    int seq_len;
    CharTokenizer& tokenizer;
    std::vector<int> tokens;
    size_t current_pos;
    
    void load_and_tokenize(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + file_path);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string text = buffer.str();
        tokens = tokenizer.encode(text);
        std::cout << "Loaded " << tokens.size() << " tokens from " << file_path << std::endl;
    }
};

void train_language_model(
    LanguageModel& model,
    DataLoader& train_loader,
    int num_epochs,
    float learning_rate,
    const std::string& checkpoint_dir = "../models/",
    int log_interval = 10
) {
    model.train();
    
    std::filesystem::create_directories(checkpoint_dir);
    
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learning_rate));
    
    std::cout << "\n=== Starting Training ===" << std::endl;
    std::cout << "Epochs: " << num_epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << "Checkpoint directory: " << checkpoint_dir << "\n" << std::endl;
    
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        train_loader.reset();
        float epoch_loss = 0.0;
        int batch_count = 0;
        
        while (train_loader.has_next()) {
            optimizer.zero_grad();
            
            auto [input, target] = train_loader.next_batch();
            
            auto logits = model.forward(input);
            
            // Compute loss: reshape for cross entropy
            // logits: [batch_size, seq_len, vocab_size]
            // target: [batch_size, seq_len]
            auto batch_size = logits.size(0);
            auto seq_len = logits.size(1);
            auto vocab_size = logits.size(2);
            
            auto logits_flat = logits.view({batch_size * seq_len, vocab_size});
            auto target_flat = target.view({batch_size * seq_len});
            
            auto loss = torch::nn::functional::cross_entropy(logits_flat, target_flat);
            
            loss.backward();
            
            // Gradient clipping
            torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
            
            optimizer.step();
            
            epoch_loss += loss.item<float>();
            batch_count++;
            
            // Log progress
            if (batch_count % log_interval == 0) {
                float avg_loss = epoch_loss / batch_count;
                std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs 
                          << "], Batch [" << batch_count 
                          << "], Loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
            }
        }
        
        float avg_epoch_loss = epoch_loss / batch_count;
        std::cout << "\nEpoch [" << epoch + 1 << "/" << num_epochs 
                  << "] completed. Average Loss: " << std::fixed << std::setprecision(4) 
                  << avg_epoch_loss << std::endl;
        
        // Save checkpoint
        std::string checkpoint_path = checkpoint_dir + "checkpoint_epoch_" + std::to_string(epoch + 1) + ".pt";
        torch::serialize::OutputArchive archive;
        model.save(archive);
        archive.save_to(checkpoint_path);
        std::cout << "Checkpoint saved to: " << checkpoint_path << "\n" << std::endl;
    }
    
    std::cout << "Training completed!" << std::endl;
}

