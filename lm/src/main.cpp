#include "lm.cpp"
#include <iostream>
#include <iomanip>
#include <filesystem>

int main() {
    std::cout << "=== Transformer Language Model Training ===" << std::endl;
    
    // Hyperparameters
    const int CONTEXT_LEN = 128;
    const int EMBEDDING_DIM = 256;
    const int NUM_HEADS = 8;
    const int NUM_LAYERS = 4;
    const float DROPOUT = 0.1;
    const int BATCH_SIZE = 16;
    const int SEQ_LEN = 128;
    const int NUM_EPOCHS = 1;
    const float LEARNING_RATE = 0.0003;
    
    // Initialize tokenizer (no EOS token)
    CharTokenizer tokenizer;
    std::cout << "Tokenizer initialized with vocab size: " << tokenizer.get_vocab_size() << std::endl;
    
    // Find training data - try multiple possible paths
    std::vector<std::string> possible_paths = {
        "lm/data/training_data.txt",           // From root directory
        "../lm/data/training_data.txt",        // From build directory
        "../../lm/data/training_data.txt",     // From build/bin directory
        "data/training_data.txt"               // If run from lm directory
    };
    
    std::string data_path;
    bool found = false;
    
    for (const auto& path : possible_paths) {
        if (std::filesystem::exists(path)) {
            data_path = path;
            found = true;
            std::cout << "Found training data at: " << path << std::endl;
            break;
        }
    }
    
    if (!found) {
        std::cerr << "\nError: Training data not found!" << std::endl;
        std::cerr << "Tried the following locations:" << std::endl;
        for (const auto& path : possible_paths) {
            std::cerr << "  - " << path << std::endl;
        }
        std::cerr << "\nPlease create training_data.txt in lm/data/ directory" << std::endl;
        std::cerr << "Current working directory: " << std::filesystem::current_path() << std::endl;
        return 1;
    }
    
    // Create data loader
    DataLoader train_loader(data_path, BATCH_SIZE, SEQ_LEN, tokenizer);
    
    // Create model with improved architecture
    LanguageModel model(tokenizer.get_vocab_size(), CONTEXT_LEN, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT);
    
    std::cout << "\n=== Model Configuration ===" << std::endl;
    std::cout << "  Vocab size: " << tokenizer.get_vocab_size() << std::endl;
    std::cout << "  Context length: " << CONTEXT_LEN << std::endl;
    std::cout << "  Embedding dim: " << EMBEDDING_DIM << std::endl;
    std::cout << "  Number of heads: " << NUM_HEADS << std::endl;
    std::cout << "  Number of layers: " << NUM_LAYERS << std::endl;
    std::cout << "  Dropout: " << DROPOUT << std::endl;

    //todo: remove later just for tests
    for (int i = 0; i<15; i++) {
    // Train the model
    train_language_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE);

    // Test generation - 100 tokens
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "=== Generation Test 1: Exactly 100 Tokens ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    model.eval();
    
    std::string prompt1 = "The";
    auto tokens1 = tokenizer.encode(prompt1);
    auto generated1 = model.generate(tokens1, 100, 0.8, tokenizer, true);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== Final Result (100 tokens) ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << tokenizer.decode(generated1) << std::endl;
    std::cout << "\nTotal tokens generated: " << (generated1.size() - tokens1.size()) << std::endl;
    
    // Test generation - 1000 tokens
    std::cout << "\n\n" << std::string(60, '=') << std::endl;
    std::cout << "=== Generation Test 2: Exactly 1000 Tokens ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::string prompt2 = "Once upon a time";
    auto tokens2 = tokenizer.encode(prompt2);
    auto generated2 = model.generate(tokens2, 1000, 0.8, tokenizer, true);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== Final Result (1000 tokens) ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << tokenizer.decode(generated2) << std::endl;
    std::cout << "\nTotal tokens generated: " << (generated2.size() - tokens2.size()) << std::endl;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "=== Training and Generation Complete ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    }
    return 0;
}