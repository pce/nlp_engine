/**
 * @file test_nlp_classifier.cpp
 * @brief Unit tests for IClassifierService and OnnxClassifier (Catch2 v3).
 *
 * All tests run without ONNX Runtime — the IOnnxService and
 * IClassifierService dependencies are satisfied by lightweight in-process
 * mocks, so this file compiles and passes even with -DDISABLE_ONNX=ON.
 *
 * Coverage
 * ────────
 *  MockOnnxService        — verifies OnnxClassifier works against any
 *                           IOnnxService implementation, not just ONNXAddon.
 *  Softmax activation     — single-label classification (e.g. SST-2 sentiment).
 *  Sigmoid activation     — multi-label classification (e.g. Toxic-BERT).
 *  Score ordering         — output is always sorted descending by score.
 *  Edge cases             — null service, unloaded service, unknown output
 *                           node name, more logits than labels.
 *  IClassifierService API — label_names(), is_loaded(), interface contract.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "../nlp/addons/onnx/classifier_service.hh"
#include "../nlp/addons/onnx/inference_result.hh"
#include "../nlp/addons/onnx/onnx_service.hh"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace pce::nlp::onnx;
using namespace pce::nlp::inference;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::Equals;

// ─────────────────────────────────────────────────────────────────────────────
// Minimal mock — satisfies IOnnxService without touching ONNX Runtime.
//
// Only infer() and is_loaded() are exercised by OnnxClassifier; the rest
// return harmless stub values so the compiler is satisfied.
// ─────────────────────────────────────────────────────────────────────────────

struct MockOnnxService final : public IOnnxService {
    InferenceResult mock_result;
    bool            loaded = true;

    // ── Constructor helpers ──────────────────────────────────────────────────

    /** Build a mock that returns a single named output tensor. */
    static std::shared_ptr<MockOnnxService>
    with_logits(const std::string& node, std::vector<float> logits, bool loaded = true) {
        auto m               = std::make_shared<MockOnnxService>();
        m->loaded            = loaded;
        m->mock_result.success      = loaded;
        m->mock_result.output_names = {node};
        m->mock_result.outputs      = {std::move(logits)};
        return m;
    }

    /** Build a mock whose infer() always reports failure. */
    static std::shared_ptr<MockOnnxService> failing() {
        auto m                 = std::make_shared<MockOnnxService>();
        m->loaded              = true;
        m->mock_result.success = false;
        m->mock_result.error   = "mock inference failure";
        return m;
    }

    // ── IOnnxService interface ───────────────────────────────────────────────

    EmbeddingResult embed(const std::string&) override {
        EmbeddingResult r;
        r.success = false;
        r.error   = "mock";
        return r;
    }

    std::vector<EmbeddingResult>
    embed_batch(const std::vector<std::string>& texts) override {
        return std::vector<EmbeddingResult>(texts.size());
    }

    float similarity(const std::string&, const std::string&) override {
        return 0.0f;
    }

    TagResult tag(const std::string&,
                  const std::vector<std::string>& = {}) override {
        TagResult r;
        r.success = false;
        r.error   = "mock";
        return r;
    }

    InferenceResult infer(const std::string&) override {
        return mock_result;
    }

    bool   is_loaded()  const noexcept override { return loaded; }
    size_t dimensions() const noexcept override { return 0; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static bool scores_are_descending(const std::vector<ClassLabel>& labels) {
    for (size_t i = 1; i < labels.size(); ++i) {
        if (labels[i].score > labels[i - 1].score) return false;
    }
    return true;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ─────────────────────────────────────────────────────────────────────────────
// Guard — null or unloaded service
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OnnxClassifier — null service returns empty result", "[classifier]") {
    OnnxClassifier clf(nullptr, {"NEGATIVE", "POSITIVE"});

    auto result = clf.classify("This is a test.");

    CHECK(result.empty());
}

TEST_CASE("OnnxClassifier — is_loaded() false when service is null", "[classifier]") {
    OnnxClassifier clf(nullptr, {"NEGATIVE", "POSITIVE"});
    CHECK_FALSE(clf.is_loaded());
}

TEST_CASE("OnnxClassifier — unloaded service returns empty result", "[classifier]") {
    auto svc = MockOnnxService::with_logits("logits", {1.0f, 3.0f}, /*loaded=*/false);
    OnnxClassifier clf(svc, {"NEGATIVE", "POSITIVE"});

    CHECK(clf.classify("Any text.").empty());
}

TEST_CASE("OnnxClassifier — is_loaded() mirrors underlying service state", "[classifier]") {
    auto loaded   = MockOnnxService::with_logits("logits", {1.0f}, true);
    auto unloaded = MockOnnxService::with_logits("logits", {1.0f}, false);

    CHECK(OnnxClassifier(loaded,   {"A"}).is_loaded());
    CHECK_FALSE(OnnxClassifier(unloaded, {"A"}).is_loaded());
}

TEST_CASE("OnnxClassifier — failing infer() returns empty result", "[classifier]") {
    OnnxClassifier clf(MockOnnxService::failing(), {"NEGATIVE", "POSITIVE"});

    CHECK(clf.classify("test").empty());
}

TEST_CASE("OnnxClassifier — unknown output node returns empty result", "[classifier]") {
    // Mock returns node "logits" but classifier asks for "output_0"
    auto svc = MockOnnxService::with_logits("logits", {1.0f, 2.0f});
    OnnxClassifier clf(svc, {"A", "B"}, "output_0");

    CHECK(clf.classify("test").empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Softmax activation (default) — single-label classification
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OnnxClassifier (softmax) — top label is highest logit", "[classifier][softmax]") {
    // POSITIVE logit (3.0) > NEGATIVE logit (1.0) → POSITIVE should win
    auto svc = MockOnnxService::with_logits("logits", {1.0f, 3.0f});
    OnnxClassifier clf(svc, {"NEGATIVE", "POSITIVE"});

    auto labels = clf.classify("Great product!");

    REQUIRE(labels.size() == 2);
    CHECK_THAT(labels[0].name, Equals("POSITIVE"));
    CHECK(labels[0].score > labels[1].score);
}

TEST_CASE("OnnxClassifier (softmax) — scores sum to approximately 1.0", "[classifier][softmax]") {
    auto svc = MockOnnxService::with_logits("logits", {1.0f, 2.0f, 0.5f});
    OnnxClassifier clf(svc, {"A", "B", "C"});

    auto labels = clf.classify("test");

    REQUIRE(labels.size() == 3);
    float total = 0.0f;
    for (const auto& l : labels) total += l.score;
    CHECK_THAT(total, WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("OnnxClassifier (softmax) — result is sorted descending", "[classifier][softmax]") {
    auto svc = MockOnnxService::with_logits("logits", {0.5f, 3.0f, 1.5f, 2.0f});
    OnnxClassifier clf(svc, {"A", "B", "C", "D"});

    auto labels = clf.classify("test");

    REQUIRE(labels.size() == 4);
    CHECK(scores_are_descending(labels));
}

TEST_CASE("OnnxClassifier (softmax) — each score is in [0, 1]", "[classifier][softmax]") {
    auto svc = MockOnnxService::with_logits("logits", {-2.0f, 0.0f, 5.0f});
    OnnxClassifier clf(svc, {"A", "B", "C"});

    for (const auto& l : clf.classify("test")) {
        CHECK(l.score >= 0.0f);
        CHECK(l.score <= 1.0f);
    }
}

TEST_CASE("OnnxClassifier (softmax) — identical logits produce equal scores", "[classifier][softmax]") {
    auto svc = MockOnnxService::with_logits("logits", {2.0f, 2.0f});
    OnnxClassifier clf(svc, {"A", "B"});

    auto labels = clf.classify("test");

    REQUIRE(labels.size() == 2);
    CHECK_THAT(labels[0].score, WithinAbs(labels[1].score, 1e-5f));
    CHECK_THAT(labels[0].score, WithinAbs(0.5f, 1e-5f));
}

TEST_CASE("OnnxClassifier (softmax) — label_names() matches constructor input", "[classifier][softmax]") {
    const std::vector<std::string> names = {"NEGATIVE", "NEUTRAL", "POSITIVE"};
    auto svc = MockOnnxService::with_logits("logits", {1.0f, 2.0f, 3.0f});
    OnnxClassifier clf(svc, names);

    CHECK(clf.label_names() == names);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sigmoid activation — independent multi-label classification
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OnnxClassifier (sigmoid) — high logit gives score near 1.0", "[classifier][sigmoid]") {
    // Large positive logit → sigmoid ≈ 1.0
    auto svc = MockOnnxService::with_logits("logits", {10.0f, -10.0f});
    OnnxClassifier clf(svc,
                       {"toxic", "safe"},
                       "logits",
                       OnnxClassifier::Activation::sigmoid);

    auto labels = clf.classify("horrible content");

    REQUIRE(labels.size() == 2);
    CHECK_THAT(labels[0].name,  Equals("toxic"));
    CHECK_THAT(labels[0].score, WithinAbs(1.0f, 1e-3f));
    CHECK_THAT(labels[1].score, WithinAbs(0.0f, 1e-3f));
}

TEST_CASE("OnnxClassifier (sigmoid) — zero logit gives score near 0.5", "[classifier][sigmoid]") {
    auto svc = MockOnnxService::with_logits("logits", {0.0f, 0.0f, 0.0f});
    OnnxClassifier clf(svc,
                       {"A", "B", "C"},
                       "logits",
                       OnnxClassifier::Activation::sigmoid);

    for (const auto& l : clf.classify("test"))
        CHECK_THAT(l.score, WithinAbs(0.5f, 1e-5f));
}

TEST_CASE("OnnxClassifier (sigmoid) — each score is in [0, 1]", "[classifier][sigmoid]") {
    auto svc = MockOnnxService::with_logits("logits", {-5.0f, 0.0f, 5.0f, 100.0f, -100.0f});
    OnnxClassifier clf(svc,
                       {"A", "B", "C", "D", "E"},
                       "logits",
                       OnnxClassifier::Activation::sigmoid);

    for (const auto& l : clf.classify("test")) {
        CHECK(l.score >= 0.0f);
        CHECK(l.score <= 1.0f);
    }
}

TEST_CASE("OnnxClassifier (sigmoid) — scores do NOT sum to 1 (independent)", "[classifier][sigmoid]") {
    // All positive logits → all scores > 0.5 → sum well above 1.0
    auto svc = MockOnnxService::with_logits("logits", {3.0f, 3.0f, 3.0f});
    OnnxClassifier clf(svc,
                       {"toxic", "obscene", "threat"},
                       "logits",
                       OnnxClassifier::Activation::sigmoid);

    auto labels = clf.classify("test");
    REQUIRE(labels.size() == 3);

    float total = 0.0f;
    for (const auto& l : labels) total += l.score;
    CHECK(total > 1.0f);
}

TEST_CASE("OnnxClassifier (sigmoid) — result is sorted descending", "[classifier][sigmoid]") {
    auto svc = MockOnnxService::with_logits("logits", {-1.0f, 3.0f, 0.5f, 2.0f});
    OnnxClassifier clf(svc,
                       {"A", "B", "C", "D"},
                       "logits",
                       OnnxClassifier::Activation::sigmoid);

    CHECK(scores_are_descending(clf.classify("test")));
}

TEST_CASE("OnnxClassifier (sigmoid) — score matches manual sigmoid calculation", "[classifier][sigmoid]") {
    const float logit = 1.5f;
    auto svc = MockOnnxService::with_logits("logits", {logit});
    OnnxClassifier clf(svc,
                       {"label"},
                       "logits",
                       OnnxClassifier::Activation::sigmoid);

    auto labels = clf.classify("test");
    REQUIRE(labels.size() == 1);
    CHECK_THAT(labels[0].score, WithinAbs(sigmoid(logit), 1e-5f));
}

// ─────────────────────────────────────────────────────────────────────────────
// Label / logit count mismatch
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OnnxClassifier — more logits than labels uses min(logits, labels)", "[classifier]") {
    // 4 logits, only 2 labels → only 2 ClassLabels produced
    auto svc = MockOnnxService::with_logits("logits", {1.0f, 5.0f, 3.0f, 2.0f});
    OnnxClassifier clf(svc, {"A", "B"});

    auto labels = clf.classify("test");
    CHECK(labels.size() == 2);
}

TEST_CASE("OnnxClassifier — more labels than logits uses min(logits, labels)", "[classifier]") {
    // 2 logits, 4 labels → only 2 ClassLabels produced
    auto svc = MockOnnxService::with_logits("logits", {1.0f, 3.0f});
    OnnxClassifier clf(svc, {"A", "B", "C", "D"});

    auto labels = clf.classify("test");
    CHECK(labels.size() == 2);
}

TEST_CASE("OnnxClassifier — empty labels vector returns empty result", "[classifier]") {
    auto svc = MockOnnxService::with_logits("logits", {1.0f, 2.0f});
    OnnxClassifier clf(svc, {});

    CHECK(clf.classify("test").empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-label edge case
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OnnxClassifier (softmax) — single label gets score 1.0", "[classifier][softmax]") {
    auto svc = MockOnnxService::with_logits("logits", {42.0f});
    OnnxClassifier clf(svc, {"ONLY"});

    auto labels = clf.classify("test");
    REQUIRE(labels.size() == 1);
    CHECK_THAT(labels[0].name,  Equals("ONLY"));
    CHECK_THAT(labels[0].score, WithinAbs(1.0f, 1e-5f));
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom output node name
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OnnxClassifier — custom output node name is respected", "[classifier]") {
    auto svc = MockOnnxService::with_logits("output_0", {1.0f, 4.0f});
    OnnxClassifier clf(svc, {"NEGATIVE", "POSITIVE"}, "output_0");

    auto labels = clf.classify("test");
    REQUIRE_FALSE(labels.empty());
    CHECK_THAT(labels[0].name, Equals("POSITIVE"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Polymorphic usage — interface contract
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OnnxClassifier — usable through IClassifierService interface", "[classifier]") {
    auto svc = MockOnnxService::with_logits("logits", {0.2f, 2.8f});
    std::shared_ptr<IClassifierService> iface =
        std::make_shared<OnnxClassifier>(svc, std::vector<std::string>{"NEG", "POS"});

    CHECK(iface->is_loaded());
    CHECK(iface->label_names().size() == 2);

    auto labels = iface->classify("I love this!");
    REQUIRE(labels.size() == 2);
    CHECK_THAT(labels[0].name, Equals("POS"));
    CHECK(scores_are_descending(labels));
}

// ─────────────────────────────────────────────────────────────────────────────
// Toxicity six-class realistic scenario
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OnnxClassifier (sigmoid) — realistic toxicity scenario", "[classifier][sigmoid]") {
    // Simulate Toxic-BERT logits: high toxic/obscene, low others
    const std::vector<float> logits = {4.0f, 2.5f, 3.1f, -1.0f, -0.5f, -2.0f};
    const std::vector<std::string> labels = {
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate"
    };

    auto svc = MockOnnxService::with_logits("logits", logits);
    OnnxClassifier clf(svc, labels, "logits", OnnxClassifier::Activation::sigmoid);

    auto result = clf.classify("test");

    REQUIRE(result.size() == 6);
    CHECK(scores_are_descending(result));

    // "toxic" has the highest logit (4.0) so should appear first
    CHECK_THAT(result[0].name, Equals("toxic"));

    // All active categories (positive logits) should score > 0.5
    float toxic_score   = sigmoid(4.0f);
    float obscene_score = sigmoid(3.1f);
    CHECK_THAT(result[0].score, WithinAbs(toxic_score,   1e-4f));

    // identity_hate has logit -2.0 → near 0
    CHECK(result.back().score < 0.2f);

    // Any score > threshold could be "positive" in multi-label sense
    const float threshold = 0.5f;
    int active_count = 0;
    for (const auto& l : result)
        if (l.score > threshold) ++active_count;
    CHECK(active_count == 3); // toxic, severe_toxic, obscene all have positive logits
}
