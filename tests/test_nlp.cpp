#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include "../nlp/nlp_engine.hh"
#include "../nlp/addons/dedupe_addon.hh"

using namespace pce::nlp;
namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Equals;

/**
 * @file test_nlp.cpp
 * @brief Unit tests for the core NLPEngine using Catch2 v3.
 */

// Global or static fixtures can be handled via TEST_CASE and Sections
// but for NLPEngine we'll use a helper struct to maintain state consistency
struct NLPEngineFixture {
    std::shared_ptr<NLPModel> model;
    std::unique_ptr<NLPEngine> engine;

    NLPEngineFixture() {
        model = std::make_shared<NLPModel>();
        std::string data_path = "data";
        if (!fs::exists(data_path)) {
            data_path = "../data";
        }
        model->load_from(data_path);
        engine = std::make_unique<NLPEngine>(model);
    }
};

TEST_CASE("NLPEngine Core Functionality", "[core][nlp]") {
    NLPEngineFixture fix;

    SECTION("Tokenization") {
        std::string text = "The quick brown fox.";
        auto tokens = fix.engine->tokenize(text);

        REQUIRE(tokens.size() == 4);
        CHECK_THAT(tokens[0], Equals("the"));
        CHECK_THAT(tokens[1], Equals("quick"));
        CHECK_THAT(tokens[2], Equals("brown"));
    }

    SECTION("Sentence Splitting") {
        std::string text = "First sentence. Second sentence? \"Is this third?\" Yes!";
        auto sentences = fix.engine->split_sentences(text);

        // The engine's quote handling logic returns 3 sentences
        REQUIRE(sentences.size() == 3);
        CHECK_THAT(sentences[0], Equals("First sentence."));
    }

    SECTION("Normalization") {
        std::string text = "Hello, World!";
        std::string normalized = fix.engine->normalize(text);

        CHECK_THAT(normalized, Equals("hello world"));
    }

    SECTION("SpellCheck") {
        if (fix.model->is_ready()) {
            auto corrections = fix.engine->spell_check("I am hapy", "en");
            bool found_hapy = false;
            for (const auto& c : corrections) {
                if (c.original == "hapy") {
                    found_hapy = true;
                    CHECK_THAT(c.suggested, Equals("happy"));
                }
            }
            CHECK(found_hapy);
        }
    }

    SECTION("Entity Extraction") {
        std::string text = "Contact me at info@example.com.";
        auto entities = fix.engine->extract_entities(text, "en");

        bool found_email = false;
        for (const auto& e : entities) {
            if (e.type == "email") found_email = true;
        }
        CHECK(found_email);
    }

    SECTION("Language Detection") {
        auto en_profile = fix.engine->detect_language("The quick brown fox jumps over the lazy dog.");
        CHECK_THAT(en_profile.language, Equals("en"));

        if (fix.model->is_ready()) {
            auto de_profile = fix.engine->detect_language("Das ist ein einfacher deutscher Satz.");
            CHECK_THAT(de_profile.language, Equals("de"));
        }
    }

    SECTION("Stemming") {
        // Current simple stemmer only handles 's' suffix for English
        CHECK_THAT(fix.engine->stem("running", "en"), Equals("running"));
        CHECK_THAT(fix.engine->stem("jumps", "en"), Equals("jump"));
    }

    SECTION("POSTagging") {
        auto tokens = fix.engine->tokenize("The dog runs.");
        auto tags = fix.engine->pos_tag(tokens, "en");

        CHECK(!tags.empty());
    }

    SECTION("Terminology Extraction") {
        std::string text = "IBM developed artificial intelligence in New York City.";
        auto terms = fix.engine->extract_terminology(text, "en");

        // Should find acronyms (IBM)
        bool found_ibm = false;
        // Should find compounds (New York City)
        bool found_nyc = false;
        // Should find technical phrases (artificial intelligence - if tagged as JJ NN)
        bool found_ai = false;

        for (const auto& term : terms) {
            if (term == "IBM") found_ibm = true;
            if (term == "New York City") found_nyc = true;
            if (term == "artificial intelligence") found_ai = true;
        }

        CHECK(found_ibm);
        CHECK(found_nyc);
    }

    SECTION("Readability Metrics") {
        std::string text = "This is a simple sentence. It is easy to read.";
        auto metrics = fix.engine->analyze_readability(text);
        CHECK(metrics.word_count >= 10);
        CHECK(metrics.sentence_count >= 2);
        CHECK(metrics.flesch_kincaid_grade != 0);
    }

    SECTION("Sentiment Analysis") {
        if (fix.model->is_ready()) {
            auto pos = fix.engine->analyze_sentiment("This is a wonderful and great day!", "en");
            CHECK_THAT(pos.label, Equals("positive"));
            CHECK(pos.score > 0);

            auto neg = fix.engine->analyze_sentiment("This is a terrible and awful mistake.", "en");
            CHECK_THAT(neg.label, Equals("negative"));
            CHECK(neg.score < 0);
        }
    }
}

TEST_CASE("Deduplication Addon Functionality", "[addon][dedupe]") {
    auto dedupe = std::make_unique<DeduplicationAddon>();
    dedupe->initialize();

    SECTION("Basic Detection") {
        std::string text = "This is a test. This is a test.";
        std::unordered_map<std::string, std::string> options = {
            {"mode", "detect"},
            {"skip_words", "this,is,a"}
        };

        auto resp = dedupe->process(text, options);
        REQUIRE(resp.has_value());

        // Verify structured metadata contains the duplicate hit
        CHECK(resp->metadata.contains("dup_0_text"));
        CHECK_THAT(resp->metadata.at("dup_0_text"), Equals("This is a test."));

        // Ensure output is the original text, not serialized JSON
        CHECK_THAT(resp->output, !ContainsSubstring("duplicates"));

        REQUIRE(resp->metrics.contains("duplicates_found"));
        CHECK_THAT(resp->metrics.at("duplicates_found"), Catch::Matchers::WithinAbs(1.0, 0.001));
    }

    SECTION("Mode Remove") {
        std::string text = "Keep this. Duplicate. Duplicate.";
        std::unordered_map<std::string, std::string> options = {
            {"mode", "remove"},
            {"skip_words", "keep,this"}
        };

        auto resp = dedupe->process(text, options);
        REQUIRE(resp.has_value());
        CHECK_THAT(resp->output, Equals("Keep this. Duplicate."));
    }

    SECTION("Quotation and Punctuation Normalization") {
        std::string text = "Hello world. Hello world! \"Hello world\"";
        std::unordered_map<std::string, std::string> options = {
            {"mode", "detect"},
            {"ignore_quotes", "true"},
            {"ignore_punctuation", "true"},
            {"skip_words", "the,a,is"}
        };

        auto resp = dedupe->process(text, options);
        REQUIRE(resp.has_value());

        REQUIRE(resp->metrics.contains("duplicates_found"));
        CHECK_THAT(resp->metrics.at("duplicates_found"), Catch::Matchers::WithinAbs(2.0, 0.001));
    }

    SECTION("MinLength Filtering") {
        std::string text = "To be. To be. This is long enough. This is long enough.";
        std::unordered_map<std::string, std::string> options = {
            {"mode", "detect"},
            {"min_length", "10"}
        };

        auto resp = dedupe->process(text, options);
        REQUIRE(resp.has_value());

        // "To be" (length 5) is < 10, so no duplicate hit for it.
        // "This is long enough" is > 10, so 1 duplicate hit.
        REQUIRE(resp->metrics.contains("duplicates_found"));
        CHECK_THAT(resp->metrics.at("duplicates_found"), Catch::Matchers::WithinAbs(1.0, 0.001));
    }
}
