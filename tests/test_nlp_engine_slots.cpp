/**
 * @file test_nlp_engine_slots.cpp
 * @brief Unit tests for NLPEngine typed service slots and NLPModel::create_empty().
 *
 * These tests exercise:
 *  - NLPModel::create_empty()      — no files, all accessors return empty, is_ready() = true
 *  - NLPEngine()                   — default constructor builds on empty model
 *  - NLPEngine::set_*_service()    — attach/detach each typed slot
 *  - NLPEngine::has_*()            — guard methods reflect the slot state
 *  - Fallback behaviour            — classical methods degrade gracefully without neural services
 *
 * No ONNX Runtime is required.  All neural-service interactions are exercised
 * through lightweight mock implementations of IOnnxService and IClassifierService
 * that live entirely in this translation unit.
 *
 * Build / run:
 *   cmake --build build --target nlp_tests_engine_slots
 *   ./build/nlp_tests_engine_slots -v
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../nlp/nlp_engine.hh"

#include <memory>
#include <string>
#include <vector>

using namespace pce::nlp;
using Catch::Matchers::Equals;
using Catch::Matchers::WithinAbs;

// ─────────────────────────────────────────────────────────────────────────────
// Minimal mocks — no ONNX Runtime header required
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Minimal IOnnxService mock.
 *
 * Controllable via the `loaded` flag.  infer() returns a pre-configured
 * InferenceResult so callers that inspect the output can be tested as well.
 */
struct MockOnnxService : public onnx::IOnnxService {
    bool loaded;

    explicit MockOnnxService(bool loaded = true) : loaded(loaded) {}

    [[nodiscard]] inference::EmbeddingResult
    embed(const std::string&) override {
        inference::EmbeddingResult r;
        r.success = false;
        r.error   = "mock — no model";
        return r;
    }

    [[nodiscard]] std::vector<inference::EmbeddingResult>
    embed_batch(const std::vector<std::string>& texts) override {
        return std::vector<inference::EmbeddingResult>(texts.size());
    }

    [[nodiscard]] float
    similarity(const std::string&, const std::string&) override {
        return 0.0f;
    }

    [[nodiscard]] inference::TagResult
    tag(const std::string&,
        const std::vector<std::string>& = {}) override {
        inference::TagResult r;
        r.success = false;
        r.error   = "mock — no model";
        return r;
    }

    [[nodiscard]] inference::InferenceResult
    infer(const std::string&) override {
        inference::InferenceResult r;
        r.success = false;
        r.error   = "mock — no model";
        return r;
    }

    [[nodiscard]] bool   is_loaded()   const noexcept override { return loaded; }
    [[nodiscard]] size_t dimensions()  const noexcept override { return 0; }
};

/**
 * @brief Minimal IClassifierService mock.
 *
 * Returns a hard-coded label list when is_loaded() is true.
 */
struct MockClassifierService : public onnx::IClassifierService {
    bool loaded;
    std::vector<std::string>      labels_;
    std::vector<onnx::ClassLabel> result_;

    explicit MockClassifierService(bool loaded = true)
        : loaded(loaded)
        , labels_{"LABEL_A", "LABEL_B"}
        , result_{{"LABEL_A", 0.9f}, {"LABEL_B", 0.1f}}
    {}

    [[nodiscard]] std::vector<onnx::ClassLabel>
    classify(const std::string&) override {
        if (!loaded) return {};
        return result_;
    }

    [[nodiscard]] const std::vector<std::string>&
    label_names() const noexcept override { return labels_; }

    [[nodiscard]] bool is_loaded() const noexcept override { return loaded; }
};

// ─────────────────────────────────────────────────────────────────────────────
// NLPModel::create_empty()
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPModel::create_empty — construction", "[model][empty]") {
    auto model = NLPModel::create_empty();

    SECTION("Returns a non-null shared_ptr") {
        REQUIRE(model != nullptr);
    }

    SECTION("is_ready() is true") {
        CHECK(model->is_ready());
    }
}

TEST_CASE("NLPModel::create_empty — accessors return empty containers", "[model][empty]") {
    auto model = NLPModel::create_empty();

    SECTION("get_stopwords(en) is empty") {
        CHECK(model->get_stopwords("en").empty());
    }

    SECTION("get_stopwords(de) is empty") {
        CHECK(model->get_stopwords("de").empty());
    }

    SECTION("get_dictionary(en) is empty") {
        CHECK(model->get_dictionary("en").empty());
    }

    SECTION("get_dictionary(fr) is empty") {
        CHECK(model->get_dictionary("fr").empty());
    }

    SECTION("get_positive_lexicon is empty") {
        CHECK(model->get_positive_lexicon().empty());
    }

    SECTION("get_negative_lexicon is empty") {
        CHECK(model->get_negative_lexicon().empty());
    }

    SECTION("get_toxic_patterns is empty") {
        CHECK(model->get_toxic_patterns().empty());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NLPEngine — default constructor
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine default constructor", "[engine][ctor]") {
    SECTION("Constructs without throwing") {
        REQUIRE_NOTHROW(NLPEngine{});
    }

    SECTION("All has_* guards are false immediately after construction") {
        NLPEngine engine;
        CHECK_FALSE(engine.has_onnx());
        CHECK_FALSE(engine.has_sentiment_model());
        CHECK_FALSE(engine.has_toxicity_model());
        CHECK_FALSE(engine.has_ner_model());
    }

    SECTION("onnx_service() returns nullptr") {
        NLPEngine engine;
        CHECK(engine.onnx_service() == nullptr);
    }
}

TEST_CASE("NLPEngine model constructor with empty model", "[engine][ctor]") {
    auto model = NLPModel::create_empty();

    SECTION("Constructs without throwing") {
        REQUIRE_NOTHROW(NLPEngine{model});
    }

    SECTION("All has_* guards are false") {
        NLPEngine engine(model);
        CHECK_FALSE(engine.has_onnx());
        CHECK_FALSE(engine.has_sentiment_model());
        CHECK_FALSE(engine.has_toxicity_model());
        CHECK_FALSE(engine.has_ner_model());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX embedding slot — set_onnx_service / has_onnx
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — ONNX embedding slot", "[engine][slots][onnx]") {
    NLPEngine engine;

    SECTION("has_onnx() false before attaching any service") {
        CHECK_FALSE(engine.has_onnx());
    }

    SECTION("has_onnx() true after attaching a loaded service") {
        auto svc = std::make_shared<MockOnnxService>(/*loaded=*/true);
        engine.set_onnx_service(svc);
        CHECK(engine.has_onnx());
    }

    SECTION("has_onnx() false when service reports not loaded") {
        auto svc = std::make_shared<MockOnnxService>(/*loaded=*/false);
        engine.set_onnx_service(svc);
        CHECK_FALSE(engine.has_onnx());
    }

    SECTION("has_onnx() false after detaching with nullptr") {
        auto svc = std::make_shared<MockOnnxService>(true);
        engine.set_onnx_service(svc);
        REQUIRE(engine.has_onnx());  // sanity
        engine.set_onnx_service(nullptr);
        CHECK_FALSE(engine.has_onnx());
    }

    SECTION("onnx_service() returns the attached service") {
        auto svc = std::make_shared<MockOnnxService>(true);
        engine.set_onnx_service(svc);
        CHECK(engine.onnx_service() == svc);
    }

    SECTION("onnx_service() returns nullptr after detach") {
        auto svc = std::make_shared<MockOnnxService>(true);
        engine.set_onnx_service(svc);
        engine.set_onnx_service(nullptr);
        CHECK(engine.onnx_service() == nullptr);
    }

    SECTION("Replacing the service updates has_onnx()") {
        auto loaded   = std::make_shared<MockOnnxService>(true);
        auto unloaded = std::make_shared<MockOnnxService>(false);

        engine.set_onnx_service(loaded);
        REQUIRE(engine.has_onnx());

        engine.set_onnx_service(unloaded);
        CHECK_FALSE(engine.has_onnx());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sentiment classifier slot — set_sentiment_service / has_sentiment_model
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — sentiment classifier slot", "[engine][slots][sentiment]") {
    NLPEngine engine;

    SECTION("has_sentiment_model() false before attaching") {
        CHECK_FALSE(engine.has_sentiment_model());
    }

    SECTION("has_sentiment_model() true with loaded classifier") {
        auto svc = std::make_shared<MockClassifierService>(/*loaded=*/true);
        engine.set_sentiment_service(svc);
        CHECK(engine.has_sentiment_model());
    }

    SECTION("has_sentiment_model() false with unloaded classifier") {
        auto svc = std::make_shared<MockClassifierService>(/*loaded=*/false);
        engine.set_sentiment_service(svc);
        CHECK_FALSE(engine.has_sentiment_model());
    }

    SECTION("has_sentiment_model() false after detach with nullptr") {
        auto svc = std::make_shared<MockClassifierService>(true);
        engine.set_sentiment_service(svc);
        REQUIRE(engine.has_sentiment_model());
        engine.set_sentiment_service(nullptr);
        CHECK_FALSE(engine.has_sentiment_model());
    }

    SECTION("Other slots unaffected by sentiment attachment") {
        auto svc = std::make_shared<MockClassifierService>(true);
        engine.set_sentiment_service(svc);
        CHECK_FALSE(engine.has_onnx());
        CHECK_FALSE(engine.has_toxicity_model());
        CHECK_FALSE(engine.has_ner_model());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Toxicity classifier slot — set_toxicity_service / has_toxicity_model
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — toxicity classifier slot", "[engine][slots][toxicity]") {
    NLPEngine engine;

    SECTION("has_toxicity_model() false before attaching") {
        CHECK_FALSE(engine.has_toxicity_model());
    }

    SECTION("has_toxicity_model() true with loaded classifier") {
        auto svc = std::make_shared<MockClassifierService>(true);
        engine.set_toxicity_service(svc);
        CHECK(engine.has_toxicity_model());
    }

    SECTION("has_toxicity_model() false with unloaded classifier") {
        auto svc = std::make_shared<MockClassifierService>(false);
        engine.set_toxicity_service(svc);
        CHECK_FALSE(engine.has_toxicity_model());
    }

    SECTION("has_toxicity_model() false after detach") {
        auto svc = std::make_shared<MockClassifierService>(true);
        engine.set_toxicity_service(svc);
        engine.set_toxicity_service(nullptr);
        CHECK_FALSE(engine.has_toxicity_model());
    }

    SECTION("Other slots unaffected by toxicity attachment") {
        auto svc = std::make_shared<MockClassifierService>(true);
        engine.set_toxicity_service(svc);
        CHECK_FALSE(engine.has_onnx());
        CHECK_FALSE(engine.has_sentiment_model());
        CHECK_FALSE(engine.has_ner_model());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NER service slot — set_ner_service / has_ner_model
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — NER service slot", "[engine][slots][ner]") {
    NLPEngine engine;

    SECTION("has_ner_model() false before attaching") {
        CHECK_FALSE(engine.has_ner_model());
    }

    SECTION("has_ner_model() true with loaded service") {
        auto svc = std::make_shared<MockOnnxService>(true);
        engine.set_ner_service(svc);
        CHECK(engine.has_ner_model());
    }

    SECTION("has_ner_model() false with unloaded service") {
        auto svc = std::make_shared<MockOnnxService>(false);
        engine.set_ner_service(svc);
        CHECK_FALSE(engine.has_ner_model());
    }

    SECTION("has_ner_model() false after detach") {
        auto svc = std::make_shared<MockOnnxService>(true);
        engine.set_ner_service(svc);
        engine.set_ner_service(nullptr);
        CHECK_FALSE(engine.has_ner_model());
    }

    SECTION("Embedding slot and NER slot are independent") {
        auto embed = std::make_shared<MockOnnxService>(true);
        auto ner   = std::make_shared<MockOnnxService>(true);

        engine.set_onnx_service(embed);
        engine.set_ner_service(ner);

        CHECK(engine.has_onnx());
        CHECK(engine.has_ner_model());

        engine.set_onnx_service(nullptr);
        CHECK_FALSE(engine.has_onnx());
        CHECK(engine.has_ner_model());  // NER slot must not be affected
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// All four slots together — isolation guarantee
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — all four service slots are independent", "[engine][slots]") {
    NLPEngine engine;

    auto onnx_svc      = std::make_shared<MockOnnxService>(true);
    auto sentiment_svc = std::make_shared<MockClassifierService>(true);
    auto toxicity_svc  = std::make_shared<MockClassifierService>(true);
    auto ner_svc       = std::make_shared<MockOnnxService>(true);

    engine.set_onnx_service(onnx_svc);
    engine.set_sentiment_service(sentiment_svc);
    engine.set_toxicity_service(toxicity_svc);
    engine.set_ner_service(ner_svc);

    REQUIRE(engine.has_onnx());
    REQUIRE(engine.has_sentiment_model());
    REQUIRE(engine.has_toxicity_model());
    REQUIRE(engine.has_ner_model());

    // Detach only one slot; remaining three must stay active
    engine.set_sentiment_service(nullptr);

    CHECK(engine.has_onnx());
    CHECK_FALSE(engine.has_sentiment_model());
    CHECK(engine.has_toxicity_model());
    CHECK(engine.has_ner_model());
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback behaviour — classical methods degrade gracefully without ONNX
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine fallback — analyze_sentiment without neural service", "[engine][fallback]") {
    // Use a loaded NLPModel so the lexicon fallback has something to work with.
    // If data/ is unavailable we fall through to empty-model path, which still
    // must not crash.
    auto model  = std::make_shared<NLPModel>();
    std::string dp = "data";
    if (!std::filesystem::exists(dp)) dp = "../data";
    model->load_from(dp);   // may or may not succeed — both are valid

    NLPEngine engine(model);
    REQUIRE_FALSE(engine.has_sentiment_model());  // guard: no neural service

    SECTION("Does not throw on English positive text") {
        SentimentResult result;
        REQUIRE_NOTHROW(result = engine.analyze_sentiment("This is wonderful!", "en"));
        // label must be a non-empty string
        CHECK_FALSE(result.label.empty());
    }

    SECTION("Does not throw on neutral / empty input") {
        SentimentResult result;
        REQUIRE_NOTHROW(result = engine.analyze_sentiment("", "en"));
        CHECK_FALSE(result.label.empty());
    }

    SECTION("score is a finite float") {
        auto result = engine.analyze_sentiment("Great service!", "en");
        CHECK(std::isfinite(result.score));
    }

    SECTION("confidence is in [0, 1]") {
        auto result = engine.analyze_sentiment("Terrible product.", "en");
        CHECK(result.confidence >= 0.0f);
        CHECK(result.confidence <= 1.0f);
    }
}

TEST_CASE("NLPEngine fallback — detect_toxicity without neural service", "[engine][fallback]") {
    auto model = std::make_shared<NLPModel>();
    std::string dp = "data";
    if (!std::filesystem::exists(dp)) dp = "../data";
    model->load_from(dp);

    NLPEngine engine(model);
    REQUIRE_FALSE(engine.has_toxicity_model());

    SECTION("Does not throw on clean text") {
        ToxicityResult result;
        REQUIRE_NOTHROW(result = engine.detect_toxicity("Have a nice day!", "en"));
        CHECK_FALSE(result.is_toxic);
    }

    SECTION("Does not throw on empty string") {
        ToxicityResult result;
        REQUIRE_NOTHROW(result = engine.detect_toxicity("", "en"));
        CHECK(result.score >= 0.0f);
    }

    SECTION("score is in [0, 1]") {
        auto result = engine.detect_toxicity("Hello world.", "en");
        CHECK(result.score >= 0.0f);
        CHECK(result.score <= 1.0f);
    }
}

TEST_CASE("NLPEngine fallback — extract_entities without NER service", "[engine][fallback]") {
    auto model = std::make_shared<NLPModel>();
    std::string dp = "data";
    if (!std::filesystem::exists(dp)) dp = "../data";
    model->load_from(dp);

    NLPEngine engine(model);
    REQUIRE_FALSE(engine.has_ner_model());

    SECTION("Does not throw") {
        std::vector<Entity> entities;
        REQUIRE_NOTHROW(
            entities = engine.extract_entities("Contact support@example.com.", "en")
        );
    }

    SECTION("Regex/heuristic fallback still finds email entities") {
        auto entities = engine.extract_entities("Contact support@example.com.", "en");
        bool found_email = false;
        for (const auto& e : entities)
            if (e.type == "email") found_email = true;
        CHECK(found_email);
    }
}

TEST_CASE("NLPEngine fallback — embed without ONNX service returns failure", "[engine][fallback]") {
    NLPEngine engine;   // empty model, no ONNX service
    REQUIRE_FALSE(engine.has_onnx());

    SECTION("Does not throw") {
        inference::EmbeddingResult result;
        REQUIRE_NOTHROW(result = engine.embed("test sentence"));
    }

    SECTION("Returns a failed result with a non-empty error") {
        auto result = engine.embed("test sentence");
        CHECK_FALSE(result.success);
        CHECK_FALSE(result.error.empty());
        CHECK(result.vector.empty());
    }
}

TEST_CASE("NLPEngine fallback — semantic_search without ONNX returns empty", "[engine][fallback]") {
    NLPEngine engine;
    REQUIRE_FALSE(engine.has_onnx());

    const std::vector<std::string> docs = {"The cat sat.", "The dog ran."};

    SECTION("Does not throw") {
        std::vector<SemanticMatch> matches;
        REQUIRE_NOTHROW(matches = engine.semantic_search("a pet", docs));
    }

    SECTION("Returns empty results") {
        auto matches = engine.semantic_search("a pet", docs);
        CHECK(matches.empty());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural override — service result is used when service is present
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("NLPEngine — sentiment neural path invoked when service present", "[engine][slots][sentiment]") {
    // Build a mock classifier that always returns POSITIVE with score 0.9
    struct PositiveMock : public onnx::IClassifierService {
        std::vector<std::string> lbl_ = {"NEGATIVE", "POSITIVE"};

        std::vector<onnx::ClassLabel> classify(const std::string&) override {
            // Return sorted descending: POSITIVE first
            return {{"POSITIVE", 0.9f}, {"NEGATIVE", 0.1f}};
        }
        const std::vector<std::string>& label_names() const noexcept override {
            return lbl_;
        }
        bool is_loaded() const noexcept override { return true; }
    };

    auto model = NLPModel::create_empty();
    NLPEngine engine(model);
    engine.set_sentiment_service(std::make_shared<PositiveMock>());

    REQUIRE(engine.has_sentiment_model());

    auto result = engine.analyze_sentiment("any text", "en");

    // The neural path should report positive
    CHECK_THAT(result.label, Equals("positive"));
    CHECK(result.score > 0.0f);
}

TEST_CASE("NLPEngine — toxicity neural path invoked when service present", "[engine][slots][toxicity]") {
    struct ToxicMock : public onnx::IClassifierService {
        std::vector<std::string> lbl_ = {"toxic", "non-toxic"};

        std::vector<onnx::ClassLabel> classify(const std::string&) override {
            return {{"toxic", 0.95f}, {"non-toxic", 0.05f}};
        }
        const std::vector<std::string>& label_names() const noexcept override {
            return lbl_;
        }
        bool is_loaded() const noexcept override { return true; }
    };

    auto model = NLPModel::create_empty();
    NLPEngine engine(model);
    engine.set_toxicity_service(std::make_shared<ToxicMock>());

    REQUIRE(engine.has_toxicity_model());

    auto result = engine.detect_toxicity("any text", "en");

    // The neural path's top label is "toxic" with score 0.95
    CHECK(result.is_toxic);
    CHECK_THAT(result.score, WithinAbs(0.95f, 0.01f));
}
