#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include "../nlp/addons/markov_addon.hh"
#include "../nlp/addons/vector_addon.hh"

using namespace pce::nlp;
namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Equals;

/**
 * @file test_nlp_markov.cpp
 * @brief Unit tests for MarkovAddon using Catch2 v3.
 *
 * This suite verifies the Markov chain generation, training pipeline,
 * and hybrid semantic filtering in a C++23 environment.
 */

struct MarkovTestFixture {
    std::unique_ptr<MarkovAddon> addon;
    const std::string test_input_path = "test_markov_input.txt";
    const std::string test_model_path = "test_markov_model.json";

    MarkovTestFixture() {
        addon = std::make_unique<MarkovAddon>();
        cleanup();
    }

    ~MarkovTestFixture() {
        cleanup();
    }

    void cleanup() {
        if (fs::exists(test_input_path)) fs::remove(test_input_path);
        if (fs::exists(test_model_path)) fs::remove(test_model_path);
    }

    void create_dummy_data(const std::string& content) {
        std::ofstream out(test_input_path);
        out << content;
        out.close();
    }
};

TEST_CASE_METHOD(MarkovTestFixture, "MarkovAddon Lifecycle and Training", "[markov][addon]") {

    SECTION("Initialization Safety") {
        // Ensure addon starts in a safe, non-ready state
        CHECK_FALSE(addon->is_ready());

        // Check error handling when model isn't loaded
        auto resp = addon->process("test", {});
        CHECK_FALSE(resp.has_value());
        CHECK_THAT(resp.error(), Equals("Markov model not loaded"));
    }

    SECTION("Training Pipeline") {
        create_dummy_data("The quick brown fox jumps over the lazy dog. The quick brown fox is fast.");

        bool success = addon->train(test_input_path, test_model_path);
        REQUIRE(success);
        REQUIRE(fs::exists(test_model_path));

        // Verify Knowledge Pack structure
        std::ifstream in(test_model_path);
        nlohmann::json j;
        in >> j;

        // The engine stores data under a "data" key in the JSON model
        auto model_data = j.contains("data") ? j["data"] : j;

        CHECK(model_data.contains("the"));
        CHECK(model_data["the"].contains("quick"));
        CHECK(model_data["the"].contains("lazy"));
    }
}

TEST_CASE_METHOD(MarkovTestFixture, "MarkovAddon Inference", "[markov][inference]") {

    SECTION("Text Generation Inference") {
        create_dummy_data("hello world hello world hello world");
        addon->train(test_input_path, test_model_path);

        bool loaded = addon->load_knowledge_pack(test_model_path);
        REQUIRE(loaded);
        REQUIRE(addon->is_ready());

        std::unordered_map<std::string, std::string> options = {
            {"length", "5"}
        };

        auto resp = addon->process("hello", options);

        REQUIRE(resp.has_value());
        CHECK(resp->output.length() > 0);

        REQUIRE(resp->metrics.contains("tokens_generated"));
        CHECK_THAT(resp->metrics.at("tokens_generated"), Catch::Matchers::WithinAbs(5.0, 0.001));
    }

    SECTION("Hybrid Vector Scoring") {
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

        {
            std::ofstream v_out(vector_model_path);
            v_out << vec_data.dump();
        }

        vector_addon->load_knowledge_pack(vector_model_path);
        addon->set_vector_engine(vector_addon);

        // 3. Generate with hybrid enabled and high threshold
        std::unordered_map<std::string, std::string> options = {
            {"length", "10"},
            {"use_hybrid", "true"},
            {"semantic_filter", "0.5"},
            {"temperature", "0.1"} // Low temperature to make it deterministic
        };

        auto resp = addon->process("apple", options);
        REQUIRE(resp.has_value());

        // With semantic penalty, "banana" should be preferred over "cherry" or "date"
        // because its similarity to "apple" (0.9) is above the threshold (0.5).
        CHECK_THAT(resp->output, ContainsSubstring("banana"));

        if (fs::exists(vector_model_path)) fs::remove(vector_model_path);
    }

    SECTION("Very Small Input (Contract Test)") {
        // Test with minimum possible data that should still work
        create_dummy_data("a b");
        addon->train(test_input_path, test_model_path);
        addon->load_knowledge_pack(test_model_path);

        // Even with just 2 words, it should be able to generate something or handle the seed
        auto resp = addon->process("a", {{"length", "1"}});
        REQUIRE(resp.has_value());
        // Since "a" is followed by "b", it should produce "b"
        CHECK_THAT(resp->output, ContainsSubstring("b"));
    }

    SECTION("Large Text Robustness") {
        std::string large_text;
        for (int i = 0; i < 100; ++i) {
            large_text += "the quick brown fox jumps over the lazy dog. ";
        }
        create_dummy_data(large_text);

        addon->train(test_input_path, test_model_path);
        bool loaded = addon->load_knowledge_pack(test_model_path);
        REQUIRE(loaded);

        auto resp = addon->process("the", {{"length", "50"}});
        REQUIRE(resp.has_value());
        CHECK(resp->output.length() > 100);

        REQUIRE(resp->metrics.contains("tokens_generated"));
        CHECK_THAT(resp->metrics.at("tokens_generated"), Catch::Matchers::WithinAbs(50.0, 0.001));
    }

    SECTION("Dead End Recovery") {
        // Create a training set where "unique" has no successor
        create_dummy_data("this is unique");
        addon->train(test_input_path, test_model_path);
        addon->load_knowledge_pack(test_model_path);

        // Requesting a long sequence starting from a dead end
        auto resp = addon->process("unique", {{"length", "10"}});

        REQUIRE(resp.has_value());
        // The engine should jump to a random word (this/is) to continue
        CHECK(resp->output.length() > 10);
    }
}
