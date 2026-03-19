#include "../nlp/addons/markov_addon.hh"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "../nlp/addons/vector_addon.hh"

namespace fs = std::filesystem;
using namespace pce::nlp;

/**
 * @file test_nlp_markov.cpp
 * @brief Professional Unit Testing for Markov Addon using CppUTest.
 *
 * This suite leverages CppUTest's memory leak detection and strict
 * harness setup/teardown for the C++23 Markov engine.
 */

TEST_GROUP(MarkovAddonTests) {
    MarkovAddon* addon;
    const std::string test_input_path = "test_markov_input.txt";
    const::std::string test_model_path = "test_markov_model.json";

    void setup() {
        // CppUTest tracks memory allocated here
        addon = new MarkovAddon();
    }

    void teardown() {
        delete addon;
        if (fs::exists(test_input_path)) fs::remove(test_input_path);
        if (fs::exists(test_model_path)) fs::remove(test_model_path);
    }

    void create_dummy_data(const std::string& content) {
        std::ofstream out(test_input_path);
        out << content;
        out.close();
    }
};

TEST(MarkovAddonTests, InitializationSafety) {
    // Ensure addon starts in a safe, non-ready state
    CHECK_FALSE(addon->is_ready());

    // Check error handling when model isn't loaded
    AddonResponse resp = addon->process("test", {});
    CHECK_FALSE(resp.success);
    STRCMP_EQUAL("Markov model not loaded", resp.error_message.c_str());
}

TEST(MarkovAddonTests, TrainingPipeline) {
    create_dummy_data("The quick brown fox jumps over the lazy dog. The quick brown fox is fast.");

    bool success = addon->train(test_input_path, test_model_path);
    CHECK_TRUE(success);
    CHECK_TRUE(fs::exists(test_model_path));

    // Verify Knowledge Pack structure
    std::ifstream in(test_model_path);
    nlohmann::json j;
    in >> j;

    CHECK_TRUE(j.contains("the"));
    CHECK_TRUE(j["the"].contains("quick"));
    CHECK_TRUE(j["the"].contains("lazy"));
}

TEST(MarkovAddonTests, TextGenerationInference) {
    create_dummy_data("hello world hello world hello world");
    addon->train(test_input_path, test_model_path);

    bool loaded = addon->load_knowledge_pack(test_model_path);
    CHECK_TRUE(loaded);
    CHECK_TRUE(addon->is_ready());

    std::unordered_map<std::string, std::string> options = {
        {"length", "5"}
    };

    AddonResponse resp = addon->process("hello", options);

    CHECK_TRUE(resp.success);
    CHECK(resp.output.length() > 0);
    DOUBLES_EQUAL(5.0, resp.metrics.at("tokens_generated"), 0.01);

    // Ensure smart punctuation/capitalization is active
    // "hello" -> "Hello" due to auto-formatting
    CHECK(std::isupper(static_cast<unsigned char>(resp.output[0])));
}

TEST(MarkovAddonTests, HybridVectorScoring) {
    // 1. Setup a simple Markov model
    create_dummy_data("apple banana apple cherry apple date");
    addon->train(test_input_path, test_model_path);
    addon->load_knowledge_pack(test_model_path);

    // 2. Setup a Vector engine with a fake "similarity"
    // We'll create a model where "apple" is more similar to "banana" than "cherry"
    auto vector_addon = std::make_shared<VectorAddon>();
    const std::string vector_model_path = "test_vector_model.json";

    nlohmann::json vec_data;
    vec_data["apple"] = {1.0, 0.0};
    vec_data["banana"] = {0.9, 0.1}; // High similarity to apple
    vec_data["cherry"] = {0.1, 0.9}; // Low similarity to apple
    vec_data["date"] = {0.0, 1.0};   // Low similarity to apple

    std::ofstream v_out(vector_model_path);
    v_out << vec_data.dump();
    v_out.close();

    vector_addon->load_knowledge_pack(vector_model_path);
    addon->set_vector_engine(vector_addon);

    // 3. Generate with hybrid enabled and high threshold
    std::unordered_map<std::string, std::string> options = {
        {"length", "10"},
        {"use_hybrid", "true"},
        {"semantic_filter", "0.5"},
        {"temperature", "0.1"} // Low temperature to make it deterministic
    };

    AddonResponse resp = addon->process("apple", options);
    CHECK_TRUE(resp.success);

    // With semantic penalty, "banana" should be preferred over "cherry" or "date"
    // because its similarity to "apple" (0.9) is above the threshold (0.5).
    // Note: The Markov counts for banana, cherry, date after "apple" are all 1 in training data.
    CHECK(resp.output.find("banana") != std::string::npos);

    if (fs::exists(vector_model_path)) fs::remove(vector_model_path);
}

TEST(MarkovAddonTests, DeadEndRecovery) {
    // Create a training set where "unique" has no successor
    create_dummy_data("this is unique");
    addon->train(test_input_path, test_model_path);
    addon->load_knowledge_pack(test_model_path);

    // Requesting a long sequence starting from a dead end
    AddonResponse resp = addon->process("unique", {{"length", "10"}});

    CHECK_TRUE(resp.success);
    // The engine should jump to a random word (this/is) to continue
    CHECK(resp.output.length() > 10);
}

int main(int ac, char** av) {
    // This runner will automatically report memory leaks
    return CommandLineTestRunner::RunAllTests(ac, av);
}
