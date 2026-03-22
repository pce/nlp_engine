/**
 * @file test_nlp_onnx_embed.cpp
 * @brief Unit tests for inference result types and ONNXAddon runtime behaviour.
 *
 * Structure:
 *   - EmbeddingResult / TagResult / InferenceResult math — always compiled,
 *     no ONNX Runtime dependency.
 *   - ONNXAddon guard tests — compiled only when NLP_WITH_ONNX is defined;
 *     model-dependent sections are skipped at runtime if no .onnx file is present.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "addons/onnx/inference_result.hh"

using namespace pce::nlp::inference;
using Catch::Matchers::WithinAbs;

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingResult — cosine similarity
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("EmbeddingResult — identical vectors have similarity 1.0", "[embed]") {
    EmbeddingResult a, b;
    a.vector  = {1.0f, 0.0f, 0.0f};
    a.success = true;
    b.vector  = {1.0f, 0.0f, 0.0f};
    b.success = true;

    CHECK_THAT(a.cosine_similarity(b), WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("EmbeddingResult — opposite vectors have similarity -1.0", "[embed]") {
    EmbeddingResult a, b;
    a.vector  = {1.0f, 0.0f};
    a.success = true;
    b.vector  = {-1.0f, 0.0f};
    b.success = true;

    CHECK_THAT(a.cosine_similarity(b), WithinAbs(-1.0f, 1e-5f));
}

TEST_CASE("EmbeddingResult — orthogonal vectors have similarity 0.0", "[embed]") {
    EmbeddingResult a, b;
    a.vector  = {1.0f, 0.0f};
    a.success = true;
    b.vector  = {0.0f, 1.0f};
    b.success = true;

    CHECK_THAT(a.cosine_similarity(b), WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("EmbeddingResult — failed result returns 0 similarity", "[embed]") {
    EmbeddingResult a, b;
    a.vector  = {1.0f, 0.0f};
    a.success = false;
    b.vector  = {1.0f, 0.0f};
    b.success = true;

    CHECK(a.cosine_similarity(b) == 0.0f);
    CHECK(b.cosine_similarity(a) == 0.0f);
}

TEST_CASE("EmbeddingResult — dimension mismatch returns 0 similarity", "[embed]") {
    EmbeddingResult a, b;
    a.vector  = {1.0f, 0.0f};
    a.success = true;
    b.vector  = {1.0f, 0.0f, 0.0f};
    b.success = true;

    CHECK(a.cosine_similarity(b) == 0.0f);
}

TEST_CASE("EmbeddingResult — similarity is clamped to [-1, 1]", "[embed]") {
    // Floating point noise can push a raw dot product slightly outside [-1, 1]
    // for non-perfectly normalised vectors; clamp must hold.
    EmbeddingResult a, b;
    a.vector  = {0.9999999f, 0.0001f};
    a.success = true;
    b.vector  = {0.9999999f, 0.0001f};
    b.success = true;

    float sim = a.cosine_similarity(b);
    CHECK(sim <= 1.0f);
    CHECK(sim >= -1.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingResult — euclidean distance
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("EmbeddingResult — same vector has euclidean distance 0", "[embed]") {
    EmbeddingResult a, b;
    a.vector  = {1.0f, 2.0f, 3.0f};
    a.success = true;
    b.vector  = {1.0f, 2.0f, 3.0f};
    b.success = true;

    CHECK_THAT(a.euclidean_distance(b), WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("EmbeddingResult — euclidean distance is symmetric", "[embed]") {
    EmbeddingResult a, b;
    a.vector  = {1.0f, 0.0f};
    a.success = true;
    b.vector  = {0.0f, 1.0f};
    b.success = true;

    CHECK_THAT(a.euclidean_distance(b), WithinAbs(b.euclidean_distance(a), 1e-5f));
}

TEST_CASE("EmbeddingResult — failed result returns max float distance", "[embed]") {
    EmbeddingResult a, b;
    a.vector  = {1.0f, 0.0f};
    a.success = false;
    b.vector  = {0.0f, 1.0f};
    b.success = true;

    CHECK(a.euclidean_distance(b) == std::numeric_limits<float>::max());
}

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingResult — serialisation
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("EmbeddingResult — to_addon_response on failure produces error response", "[embed]") {
    EmbeddingResult r;
    r.success = false;
    r.error   = "test error";

    auto resp = r.to_addon_response();
    CHECK_FALSE(resp.success);
    CHECK(resp.output.empty());
}

TEST_CASE("EmbeddingResult — to_addon_response on success contains vector", "[embed]") {
    EmbeddingResult r;
    r.success    = true;
    r.input_text = "hello";
    r.dimensions = 3;
    r.vector     = {0.1f, 0.2f, 0.3f};

    auto resp = r.to_addon_response();
    CHECK(resp.success);
    CHECK_FALSE(resp.output.empty());
    CHECK(resp.metrics.at("dimensions") == 3.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// TagResult — entity extraction and filtering
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("TagResult — entities() merges consecutive B-/I- spans", "[tag]") {
    TagResult result;
    result.success = true;
    result.tags    = {
        {"NASA",   "B-ORG", 0.95f, 0},
        {"and",    "O",     0.99f, 5},
        {"Space",  "B-ORG", 0.91f, 9},
        {"X",      "I-ORG", 0.88f, 15},
    };

    auto ents = result.entities();
    REQUIRE(ents.count("ORG") == 1);
    CHECK(ents.at("ORG").size() == 2);
    CHECK(ents.at("ORG")[0] == "NASA");
    CHECK(ents.at("ORG")[1] == "Space X");
}

TEST_CASE("TagResult — filter() returns only matching label prefix", "[tag]") {
    TagResult result;
    result.success = true;
    result.tags    = {
        {"Paris", "B-LOC", 0.97f, 0},
        {"is",    "O",     0.99f, 6},
        {"NASA",  "B-ORG", 0.95f, 9},
    };

    auto locs = result.filter("B-LOC");
    REQUIRE(locs.size() == 1);
    CHECK(locs[0].token == "Paris");

    auto other = result.filter("O");
    CHECK(other.size() == 1);
}

TEST_CASE("TagResult — to_addon_response on failure produces error response", "[tag]") {
    TagResult r;
    r.success = false;
    r.error   = "no logits";

    auto resp = r.to_addon_response();
    CHECK_FALSE(resp.success);
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceResult — argmax and softmax
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("InferenceResult — argmax returns index of highest logit", "[infer]") {
    InferenceResult r;
    r.success      = true;
    r.output_names = {"logits"};
    r.outputs      = {{0.1f, 0.9f, 0.3f}};

    CHECK(r.argmax("logits") == 1);
}

TEST_CASE("InferenceResult — softmax sums to 1.0", "[infer]") {
    InferenceResult r;
    r.success      = true;
    r.output_names = {"logits"};
    r.outputs      = {{1.0f, 2.0f, 3.0f}};

    auto probs = r.softmax("logits");
    REQUIRE(probs.size() == 3);

    float sum = 0.0f;
    for (float p : probs) sum += p;
    CHECK_THAT(sum, WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("InferenceResult — get() returns nullptr for unknown name", "[infer]") {
    InferenceResult r;
    r.success      = true;
    r.output_names = {"logits"};
    r.outputs      = {{1.0f, 2.0f}};

    CHECK(r.get("logits") != nullptr);
    CHECK(r.get("nonexistent") == nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNXAddon — guard behaviour (no model loaded)
// Requires NLP_WITH_ONNX; skipped entirely in DISABLE_ONNX builds.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef NLP_WITH_ONNX
#include "addons/onnx_addon.hh"

TEST_CASE("ONNXAddon — is_loaded() false before load_model()", "[addon][onnx]") {
    pce::nlp::onnx::ONNXAddon addon;
    CHECK_FALSE(addon.is_loaded());
    CHECK(addon.dimensions() == 0);
    CHECK(addon.vocab_size() == 0);
}

TEST_CASE("ONNXAddon — embed() returns failure when not loaded", "[addon][onnx]") {
    pce::nlp::onnx::ONNXAddon addon;
    auto r = addon.embed("test sentence");

    CHECK_FALSE(r.success);
    CHECK_FALSE(r.error.empty());
    CHECK(r.vector.empty());
}

TEST_CASE("ONNXAddon — embed_batch() returns per-item failures when not loaded", "[addon][onnx]") {
    pce::nlp::onnx::ONNXAddon addon;
    auto results = addon.embed_batch({"a", "b", "c"});

    REQUIRE(results.size() == 3);
    for (const auto& r : results)
        CHECK_FALSE(r.success);
}

TEST_CASE("ONNXAddon — similarity() returns 0 when not loaded", "[addon][onnx]") {
    pce::nlp::onnx::ONNXAddon addon;
    CHECK(addon.similarity("hello", "world") == 0.0f);
}

TEST_CASE("ONNXAddon — tag() returns failure when not loaded", "[addon][onnx]") {
    pce::nlp::onnx::ONNXAddon addon;
    auto r = addon.tag("NASA launched a rocket.");

    CHECK_FALSE(r.success);
    CHECK_FALSE(r.error.empty());
}

TEST_CASE("ONNXAddon — infer() returns failure when not loaded", "[addon][onnx]") {
    pce::nlp::onnx::ONNXAddon addon;
    auto r = addon.infer("test");

    CHECK_FALSE(r.success);
    CHECK_FALSE(r.error.empty());
}

TEST_CASE("ONNXAddon — process() returns unexpected when not loaded", "[addon][onnx]") {
    pce::nlp::onnx::ONNXAddon addon;
    auto result = addon.process("hello", {});

    CHECK_FALSE(result.has_value());
}

TEST_CASE("ONNXAddon — load_model() with nonexistent path returns false", "[addon][onnx]") {
    pce::nlp::onnx::ONNXAddon addon;
    bool loaded = addon.load_model("/nonexistent/path/model.onnx");

    CHECK_FALSE(loaded);
    CHECK_FALSE(addon.is_loaded());
}

TEST_CASE("ONNXAddon — implements IOnnxService interface", "[addon][onnx]") {
    // Verify the addon can be used polymorphically through the interface
    auto addon = std::make_shared<pce::nlp::onnx::ONNXAddon>();
    std::shared_ptr<pce::nlp::onnx::IOnnxService> svc = addon;

    CHECK_FALSE(svc->is_loaded());
    CHECK(svc->dimensions() == 0);
}

#endif  // NLP_WITH_ONNX
