#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include "../nlp/addons/markov_addon.hh"

namespace fs = std::filesystem;
using namespace pce::nlp;

/**
 * @file test_nlp_markov.cpp
 * @brief Unit tests for the Markov Addon and its training pipeline.
 */

void test_markov_training() {
    std::cout << "Running: test_markov_training..." << std::endl;

    // 1. Prepare dummy training data
    std::string test_data_path = "test_input.txt";
    std::string model_output_path = "test_model.json";

    std::ofstream out(test_data_path);
    out << "The quick brown fox jumps over the lazy dog. The quick brown fox is fast.";
    out.close();

    // 2. Train
    MarkovAddon addon;
    bool success = addon.train(test_data_path, model_output_path);
    assert(success == true);
    assert(fs::exists(model_output_path));

    // 3. Verify content
    std::ifstream in(model_output_path);
    nlohmann::json j;
    in >> j;

    // "the" should have "quick" and "lazy" as following words
    assert(j.contains("the"));
    assert(j["the"].contains("quick"));
    assert(j["the"].contains("lazy"));

    std::cout << "test_markov_training passed." << std::endl;

    // Cleanup
    fs::remove(test_data_path);
}

void test_markov_generation() {
    std::cout << "Running: test_markov_generation..." << std::endl;

    std::string model_path = "test_model.json";

    MarkovAddon addon;
    // Load the model trained in the previous step or ensure it exists
    if (!fs::exists(model_path)) {
        std::ofstream out("temp.txt");
        out << "hello world hello world hello world";
        out.close();
        addon.train("temp.txt", model_path);
        fs::remove("temp.txt");
    }

    bool loaded = addon.load_knowledge_pack(model_path);
    assert(loaded == true);
    assert(addon.is_ready() == true);

    // Test generation
    std::unordered_map<std::string, std::string> options = {
        {"length", "5"},
        {"temperature", "1.0"}
    };

    AddonResponse resp = addon.process("hello", options);

    assert(resp.success == true);
    assert(!resp.output.empty());
    assert(resp.metrics.at("tokens_generated") == 5.0);

    // Check if the output actually starts with the seed (case-insensitive)
    std::string output_lower = resp.output;
    for (auto& c : output_lower) c = std::tolower(c);
    assert(output_lower.find("hello") == 0);

    std::cout << "test_markov_generation passed." << std::endl;

    // Cleanup
    fs::remove(model_path);
}

void test_markov_empty_handling() {
    std::cout << "Running: test_markov_empty_handling..." << std::endl;

    MarkovAddon addon;
    assert(addon.is_ready() == false);

    AddonResponse resp = addon.process("test", {});
    assert(resp.success == false);
    assert(resp.error_message == "Markov model not loaded");

    std::cout << "test_markov_empty_handling passed." << std::endl;
}

int main() {
    try {
        test_markov_training();
        test_markov_generation();
        test_markov_empty_handling();

        std::cout << "\nALL MARKOV TESTS PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
