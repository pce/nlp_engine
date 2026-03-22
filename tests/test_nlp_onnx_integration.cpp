/**
 * @file test_nlp_onnx_integration.cpp
 * @brief Integration tests for ONNXAddon with a real loaded model.
 *
 * Requires all-MiniLM-L6-v2.onnx to be present in data/models/.
 * Run the download script first:
 *
 *   ./scripts/download_models.sh      (bash)
 *   .\scripts\download_models.ps1     (PowerShell)
 *   python scripts/download_models.py (Python)
 *
 * All tests skip gracefully when the model file is not found so that
 * CI environments without the model do not fail.
 */

#ifdef NLP_WITH_ONNX

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "addons/onnx_addon.hh"

#include <filesystem>
#include <numeric>

using namespace pce::nlp::onnx;
using Catch::Matchers::WithinAbs;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::filesystem::path models_dir() {
    for (auto candidate : {"data/models", "../data/models"}) {
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return {};
}

static ONNXAddon::Config make_config() {
    const auto dir   = models_dir();
    const auto model = dir / "all-MiniLM-L6-v2.onnx";
    const auto vocab = dir / "all-MiniLM-L6-v2.vocab.txt";

    ONNXAddon::Config cfg;
    cfg.model_path      = model;
    cfg.vocab_path      = vocab;
    cfg.use_mean_pooling = true;
    return cfg;
}

static float l2_norm(const std::vector<float>& v) {
    float sq = 0.0f;
    for (float x : v) sq += x * x;
    return std::sqrt(sq);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture
// ─────────────────────────────────────────────────────────────────────────────

struct ONNXFixture {
    ONNXAddon addon;
    bool      ready = false;

    ONNXFixture() {
        const auto dir = models_dir();
        if (dir.empty() || !std::filesystem::exists(dir / "all-MiniLM-L6-v2.onnx")) return;
        ready = addon.load_model(make_config());
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Load
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — load model", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    CHECK(addon.is_loaded());
    CHECK(addon.dimensions() == 384);
}

// ─────────────────────────────────────────────────────────────────────────────
// Embed — single sentence
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — embed() produces valid vector", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    auto r = addon.embed("The quick brown fox jumps over the lazy dog.");

    REQUIRE(r.success);
    CHECK(r.dimensions == 384);
    CHECK(r.vector.size() == 384);
    CHECK_FALSE(r.error.empty() == false);  // no error

    // Vector must be L2-normalised — norm should be ≈ 1.0
    CHECK_THAT(l2_norm(r.vector), WithinAbs(1.0f, 1e-4f));
}

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — embed() empty string does not crash", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    auto r = addon.embed("");
    // Empty input produces CLS+SEP only — should still return a valid (if trivial) vector
    CHECK(r.dimensions == 384);
}

// ─────────────────────────────────────────────────────────────────────────────
// Embed batch
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — embed_batch() returns one result per input", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    const std::vector<std::string> sentences = {
        "NASA launched a new rocket.",
        "The cat sat on the mat.",
        "Stock markets fell sharply on Tuesday."
    };

    auto results = addon.embed_batch(sentences);

    REQUIRE(results.size() == 3);
    for (const auto& r : results) {
        CHECK(r.success);
        CHECK(r.vector.size() == 384);
        CHECK_THAT(l2_norm(r.vector), WithinAbs(1.0f, 1e-4f));
    }
}

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — embed_batch() preserves input order", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    const std::vector<std::string> sentences = {"first sentence", "second sentence"};
    auto results = addon.embed_batch(sentences);

    REQUIRE(results.size() == 2);
    CHECK(results[0].input_text == "first sentence");
    CHECK(results[1].input_text == "second sentence");
}

// ─────────────────────────────────────────────────────────────────────────────
// Similarity
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — similarity() is symmetric", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    const std::string a = "A dog played in the park.";
    const std::string b = "A puppy ran outside.";

    float ab = addon.similarity(a, b);
    float ba = addon.similarity(b, a);

    CHECK_THAT(ab, WithinAbs(ba, 1e-4f));
}

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — similarity() ranks related > unrelated", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    const std::string query    = "A dog ran in the park.";
    const std::string related  = "A puppy was playing outside.";
    const std::string unrelated = "The central bank raised interest rates.";

    float sim_related   = addon.similarity(query, related);
    float sim_unrelated = addon.similarity(query, unrelated);

    CHECK(sim_related > sim_unrelated);
}

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — similarity() of identical sentences is near 1.0", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    const std::string s = "Artificial intelligence is transforming the world.";
    float sim = addon.similarity(s, s);

    CHECK_THAT(sim, WithinAbs(1.0f, 1e-4f));
}

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — similarity() is in [-1, 1]", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    float sim = addon.similarity("hello world", "goodbye moon");

    CHECK(sim >= -1.0f);
    CHECK(sim <=  1.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// IOnnxService — polymorphic usage
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ONNXAddon — usable through IOnnxService interface", "[onnx][integration]") {
    const auto dir = models_dir();
    if (dir.empty() || !std::filesystem::exists(dir / "all-MiniLM-L6-v2.onnx"))
        SKIP("Model not found — run scripts/download_models.sh");

    auto concrete = std::make_shared<ONNXAddon>();
    REQUIRE(concrete->load_model(make_config()));

    // Use exclusively through the abstract interface from here on
    std::shared_ptr<IOnnxService> svc = concrete;

    CHECK(svc->is_loaded());
    CHECK(svc->dimensions() == 384);

    auto r = svc->embed("language models encode meaning as geometry");
    CHECK(r.success);
    CHECK(r.vector.size() == 384);
    CHECK_THAT(l2_norm(r.vector), WithinAbs(1.0f, 1e-4f));
}

// ─────────────────────────────────────────────────────────────────────────────
// process_impl — dispatch via NLPAddon interface
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — process() embed method returns JSON with vector", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    auto result = addon.process("semantic meaning lives in geometry", {{"method", "embed"}});

    REQUIRE(result.has_value());
    CHECK(result->success);
    CHECK_FALSE(result->output.empty());
}

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — process() similarity method returns score", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    auto result = addon.process(
        "A dog ran in the park.",
        {{"method", "similarity"}, {"target", "A puppy played outside."}});

    REQUIRE(result.has_value());
    CHECK(result->success);
    CHECK(result->metrics.count("cosine_similarity") == 1);
    CHECK(result->metrics.at("cosine_similarity") > 0.0);
}

TEST_CASE_METHOD(ONNXFixture, "ONNXAddon — process() batch method returns array", "[onnx][integration]") {
    if (!ready) SKIP("Model not found — run scripts/download_models.sh");

    auto result = addon.process(
        "first sentence\nsecond sentence\nthird sentence",
        {{"method", "batch"}});

    REQUIRE(result.has_value());
    CHECK(result->success);
    CHECK(result->metrics.at("batch_size") == 3.0);
}

#endif  // NLP_WITH_ONNX
