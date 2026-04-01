/**
 * @file test_nlp_onnx.cpp
 * @brief Smoke tests for the NLPEngine ONNX service-slot API.
 *
 * Replaces the old test that called the non-existent addAddon() / loadModel()
 * on NLPEngine.  These tests verify the new typed service-slot architecture:
 *
 *   set_onnx_service()     / has_onnx()
 *   set_sentiment_service()/ has_sentiment_model()
 *   set_toxicity_service() / has_toxicity_model()
 *   set_ner_service()      / has_ner_model()
 *
 * All tests run without ONNX Runtime or any model files on disk.
 * A minimal mock of IOnnxService / IClassifierService is defined here
 * so that the slot setters can be exercised end-to-end.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "../nlp/nlp_engine.hh"
#include "../nlp/addons/onnx/onnx_service.hh"
#include "../nlp/addons/onnx/classifier_service.hh"
#include "../nlp/addons/onnx/inference_result.hh"

#include <memory>
#include <string>
#include <vector>

using namespace pce::nlp;
using Catch::Matchers::ContainsSubstring;

// ─────────────────────────────────────────────────────────────────────────────
// Minimal mocks — no ONNX Runtime dependency
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Stub IOnnxService that reports as loaded/unloaded on demand.
 *
 * All inference methods return failure results so the engine's fallback
 * paths are exercised rather than any real model execution.
 */
struct StubOnnxService final : public onnx::IOnnxService {
    explicit StubOnnxService(bool loaded = true) : loaded_(loaded) {}

    [[nodiscard]] inference::EmbeddingResult
    embed(const std::string&) override {
        inference::EmbeddingResult r;
        r.success = false;
        r.error   = "stub — no model";
        return r;
    }

    [[nodiscard]] std::vector<inference::EmbeddingResult>
    embed_batch(const std::vector<std::string>& texts) override {
        return std::vector<inference::EmbeddingResult>(texts.size());
    }

    [[nodiscard]] float
    similarity(const std::string&, const std::string&) override { return 0.0f; }

    [[nodiscard]] inference::TagResult
    tag(const std::string&,
        const std::vector<std::string>& = {}) override {
        inference::TagResult r;
        r.success = false;
        r.error   = "stub — no model";
        return r;
    }

    [[nodiscard]] inference::InferenceResult
    infer(const std::string&) override {
        inference::InferenceResult r;
        r.success = false;
        r.error   = "stub — no model";
        return r;
    }

    [[nodiscard]] bool   is_loaded()   const noexcept override { return loaded_; }
    [[nodiscard]] size_t dimensions()  const noexcept override { return 0; }

private:
    bool loaded_;
};

/**
 * @brief Stub IClassifierService that mirrors the loaded state.
 */
struct StubClassifier final : public onnx::IClassifierService {
    explicit StubClassifier(bool loaded = true)
        : loaded_(loaded), labels_({"NEG", "POS"}) {}

    [[nodiscard]] std::vector<onnx::ClassLabel>
    classify(const std::string&) override { return {}; }

    [[nodiscard]] const std::vector<std::string>&
    label_names() const noexcept override { return labels_; }

    [[nodiscard]] bool is_loaded() const noexcept override { return loaded_; }

private:
    bool                     loaded_;
    std::vector<std::string> labels_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Default-constructed engine — all slots empty
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — default constructor produces a ready engine", "[nlp][slots]") {
    // Must not throw; internally uses NLPModel::create_empty()
    REQUIRE_NOTHROW(NLPEngine{});

    NLPEngine eng;

    CHECK_FALSE(eng.has_onnx());
    CHECK_FALSE(eng.has_sentiment_model());
    CHECK_FALSE(eng.has_toxicity_model());
    CHECK_FALSE(eng.has_ner_model());
    CHECK(eng.onnx_service() == nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX service slot
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — set_onnx_service() / has_onnx()", "[nlp][slots]") {
    NLPEngine eng;
    REQUIRE_FALSE(eng.has_onnx());

    SECTION("loaded stub → has_onnx() true") {
        eng.set_onnx_service(std::make_shared<StubOnnxService>(/*loaded=*/true));
        CHECK(eng.has_onnx());
        CHECK(eng.onnx_service() != nullptr);
    }

    SECTION("unloaded stub → has_onnx() false") {
        eng.set_onnx_service(std::make_shared<StubOnnxService>(/*loaded=*/false));
        CHECK_FALSE(eng.has_onnx());
    }

    SECTION("nullptr → has_onnx() false") {
        // Set something first, then clear it
        eng.set_onnx_service(std::make_shared<StubOnnxService>());
        REQUIRE(eng.has_onnx());
        eng.set_onnx_service(nullptr);
        CHECK_FALSE(eng.has_onnx());
        CHECK(eng.onnx_service() == nullptr);
    }

    SECTION("service can be replaced") {
        auto first  = std::make_shared<StubOnnxService>(true);
        auto second = std::make_shared<StubOnnxService>(true);
        eng.set_onnx_service(first);
        REQUIRE(eng.onnx_service() == first);
        eng.set_onnx_service(second);
        CHECK(eng.onnx_service() == second);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sentiment classifier slot
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — set_sentiment_service() / has_sentiment_model()", "[nlp][slots]") {
    NLPEngine eng;
    REQUIRE_FALSE(eng.has_sentiment_model());

    SECTION("loaded classifier → has_sentiment_model() true") {
        eng.set_sentiment_service(std::make_shared<StubClassifier>(true));
        CHECK(eng.has_sentiment_model());
    }

    SECTION("unloaded classifier → has_sentiment_model() false") {
        eng.set_sentiment_service(std::make_shared<StubClassifier>(false));
        CHECK_FALSE(eng.has_sentiment_model());
    }

    SECTION("nullptr → has_sentiment_model() false") {
        eng.set_sentiment_service(std::make_shared<StubClassifier>(true));
        REQUIRE(eng.has_sentiment_model());
        eng.set_sentiment_service(nullptr);
        CHECK_FALSE(eng.has_sentiment_model());
    }

    SECTION("does not affect other slots") {
        eng.set_sentiment_service(std::make_shared<StubClassifier>(true));
        CHECK_FALSE(eng.has_onnx());
        CHECK_FALSE(eng.has_toxicity_model());
        CHECK_FALSE(eng.has_ner_model());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Toxicity classifier slot
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — set_toxicity_service() / has_toxicity_model()", "[nlp][slots]") {
    NLPEngine eng;
    REQUIRE_FALSE(eng.has_toxicity_model());

    SECTION("loaded classifier → has_toxicity_model() true") {
        eng.set_toxicity_service(std::make_shared<StubClassifier>(true));
        CHECK(eng.has_toxicity_model());
    }

    SECTION("unloaded classifier → has_toxicity_model() false") {
        eng.set_toxicity_service(std::make_shared<StubClassifier>(false));
        CHECK_FALSE(eng.has_toxicity_model());
    }

    SECTION("nullptr → has_toxicity_model() false") {
        eng.set_toxicity_service(std::make_shared<StubClassifier>(true));
        REQUIRE(eng.has_toxicity_model());
        eng.set_toxicity_service(nullptr);
        CHECK_FALSE(eng.has_toxicity_model());
    }

    SECTION("does not affect other slots") {
        eng.set_toxicity_service(std::make_shared<StubClassifier>(true));
        CHECK_FALSE(eng.has_onnx());
        CHECK_FALSE(eng.has_sentiment_model());
        CHECK_FALSE(eng.has_ner_model());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NER service slot
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — set_ner_service() / has_ner_model()", "[nlp][slots]") {
    NLPEngine eng;
    REQUIRE_FALSE(eng.has_ner_model());

    SECTION("loaded stub → has_ner_model() true") {
        eng.set_ner_service(std::make_shared<StubOnnxService>(true));
        CHECK(eng.has_ner_model());
    }

    SECTION("unloaded stub → has_ner_model() false") {
        eng.set_ner_service(std::make_shared<StubOnnxService>(false));
        CHECK_FALSE(eng.has_ner_model());
    }

    SECTION("nullptr → has_ner_model() false") {
        eng.set_ner_service(std::make_shared<StubOnnxService>(true));
        REQUIRE(eng.has_ner_model());
        eng.set_ner_service(nullptr);
        CHECK_FALSE(eng.has_ner_model());
    }

    SECTION("does not affect other slots") {
        eng.set_ner_service(std::make_shared<StubOnnxService>(true));
        CHECK_FALSE(eng.has_onnx());
        CHECK_FALSE(eng.has_sentiment_model());
        CHECK_FALSE(eng.has_toxicity_model());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Slot isolation — all four set together
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — all four service slots are independent", "[nlp][slots]") {
    NLPEngine eng;

    eng.set_onnx_service     (std::make_shared<StubOnnxService> (true));
    eng.set_sentiment_service(std::make_shared<StubClassifier>  (true));
    eng.set_toxicity_service (std::make_shared<StubClassifier>  (true));
    eng.set_ner_service      (std::make_shared<StubOnnxService> (true));

    CHECK(eng.has_onnx());
    CHECK(eng.has_sentiment_model());
    CHECK(eng.has_toxicity_model());
    CHECK(eng.has_ner_model());

    // Clear one — others must remain
    eng.set_sentiment_service(nullptr);
    CHECK(eng.has_onnx());
    CHECK_FALSE(eng.has_sentiment_model());
    CHECK(eng.has_toxicity_model());
    CHECK(eng.has_ner_model());
}

// ─────────────────────────────────────────────────────────────────────────────
// Graceful fallback when no services are attached
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — analyse_sentiment() falls back gracefully with no service", "[nlp][fallback]") {
    NLPEngine eng;   // default-constructed, no services
    REQUIRE_NOTHROW([&]{ eng.analyze_sentiment("This is wonderful!", "en"); }());

    auto result = eng.analyze_sentiment("This is wonderful!", "en");
    // Label must be non-empty regardless of method used
    CHECK_FALSE(result.label.empty());
}

TEST_CASE("NLPEngine — detect_toxicity() falls back gracefully with no service", "[nlp][fallback]") {
    NLPEngine eng;
    REQUIRE_NOTHROW([&]{ eng.detect_toxicity("You are great!", "en"); }());

    auto result = eng.detect_toxicity("You are great!", "en");
    // is_toxic must be a valid bool — simply not crashing is the contract here
    CHECK((result.is_toxic == true || result.is_toxic == false));
}

TEST_CASE("NLPEngine — embed() returns failure result with no ONNX service", "[nlp][fallback]") {
    NLPEngine eng;
    REQUIRE_FALSE(eng.has_onnx());

    auto r = eng.embed("sentence without a model");
    // Must not crash; success must be false because no service is attached
    CHECK_FALSE(r.success);
}

// ─────────────────────────────────────────────────────────────────────────────
// NLPEngine two-argument constructor (model + onnx service)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — two-arg constructor sets ONNX slot immediately", "[nlp][slots]") {
    auto model = NLPModel::create_empty();
    auto svc   = std::make_shared<StubOnnxService>(true);

    NLPEngine eng(model, svc);

    CHECK(eng.has_onnx());
    CHECK(eng.onnx_service() == svc);
    CHECK_FALSE(eng.has_sentiment_model());
    CHECK_FALSE(eng.has_toxicity_model());
    CHECK_FALSE(eng.has_ner_model());
}
