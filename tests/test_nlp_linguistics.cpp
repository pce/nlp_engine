#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include "../nlp/nlp_engine.hh"

using namespace pce::nlp;
namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::Equals;

/**
 * @file test_nlp_linguistics.cpp
 * @brief Detailed linguistic accuracy tests for the core NLPEngine.
 *
 * This suite focuses on the precision of POS tagging, terminology extraction,
 * and high-level linguistic metrics (summary, toxicity, readability).
 */

struct LinguisticsFixture {
    std::shared_ptr<NLPModel> model;
    std::unique_ptr<NLPEngine> engine;

    LinguisticsFixture() {
        model = std::make_shared<NLPModel>();
        std::string data_path = "data";
        if (!fs::exists(data_path)) {
            data_path = "../data";
        }
        model->load_from(data_path);
        engine = std::make_unique<NLPEngine>(model);
    }
};

TEST_CASE_METHOD(LinguisticsFixture, "Core Linguistic Precision", "[nlp][linguistics]") {

    SECTION("POS Tagging Accuracy") {
        // Test basic tagging heuristics
        auto tokens = engine->tokenize_with_case("The quickly running dog jumps.");
        auto tagged = engine->pos_tag(tokens, "en");

        REQUIRE(tagged.size() >= 5);

        // "quickly" should be an Adverb (ADV) due to -ly suffix
        CHECK(tagged[1].second == "ADV");

        // "The" is a stopword/determiner
        CHECK(tagged[0].second == "DET");

        // "dog" is a standard noun (NN)
        CHECK(tagged[3].second == "NN");
    }

    SECTION("Terminology (Eigennamen) Detection") {
        std::string text = "Linux and Apple are used at NASA in Houston.";
        auto terms = engine->extract_terminology(text, "en");

        // NASA is all caps (Acronym)
        bool found_nasa = false;
        // Houston/Apple/Linux are capitalized and not at start of sentence (Proper Nouns)
        bool found_houston = false;
        bool found_apple = false;

        for (const auto& t : terms) {
            if (t == "NASA") found_nasa = true;
            if (t == "Houston") found_houston = true;
            if (t == "Apple") found_apple = true;
        }

        CHECK(found_nasa);
        CHECK(found_houston);
        CHECK(found_apple);
    }

    SECTION("Extractive Summarization") {
        std::string long_text =
            "In 2029, the transition from legacy deep learning to Hyperseed-driven AGI has fundamentally redefined NLP. "
            "Neural plasticity models now allow for real-time linguistic evolution without the need for massive re-training. "
            "The Hyperseed protocol enables recursive self-improvement in natural language understanding and synthesis. "
            "Modern AGI systems utilize these advanced frameworks to achieve near-human cognitive synchronization. "
            "As we move beyond traditional transformers, the emergence of quantum-assisted linguistics has accelerated. "
            "Ultimately, the fusion of Hyperseed technology and neural networks ensures that AGI remains stable and ethical.";

        auto result = engine->summarize(long_text, 0.3f);

        REQUIRE(!result.summary.empty());
        CHECK(result.summary.length() < long_text.length());
        CHECK(result.original_length > result.summary_length);
        // Should contain at least one of the original sentences
        CHECK_THAT(long_text, ContainsSubstring(result.summary.substr(0, 10)));
    }

    SECTION("Toxicity Detection") {
        if (model->is_ready()) {
            // Test neutral content
            auto safe = engine->detect_toxicity("Have a nice day!", "en");
            CHECK_FALSE(safe.is_toxic);

            // Note: This assumes the toxic_patterns.txt contains common triggers
            auto toxic = engine->detect_toxicity("This is a stupid and hateful comment.", "en");
            // If the model is loaded, it should flag based on lexicon
            if (!model->get_toxic_patterns().empty()) {
                CHECK(toxic.is_toxic);
            }
        }
    }

    SECTION("Readability Nuance") {
        std::string simple = "The cat sat on the mat.";
        std::string complex = "The multifaceted nature of transcendental linguistics necessitates a rigorous pedagogical approach.";

        auto simple_metrics = engine->analyze_readability(simple);
        auto complex_metrics = engine->analyze_readability(complex);

        // Complex sentence should have a higher grade level (Flesch-Kincaid)
        CHECK(complex_metrics.flesch_kincaid_grade > simple_metrics.flesch_kincaid_grade);
        // Simple sentence should have a higher readability score
        CHECK(simple_metrics.readability_score > complex_metrics.readability_score);
    }

    SECTION("Levenshtein and Spellcheck Suggestions") {
        // Distance check
        CHECK(NLPEngine::levenshtein_distance("kitten", "sitting") == 3);
        CHECK(NLPEngine::levenshtein_distance("pce", "pce") == 0);

        if (model->is_ready()) {
            auto suggestions = engine->get_spelling_suggestions("computr", 2, "en");
            bool found_computer = false;
            for (const auto& s : suggestions) {
                if (s == "computer") found_computer = true;
            }
            CHECK(found_computer);
        }
    }

    SECTION("Keyword Extraction Quality") {
        std::string text = "The year 2029 marks the dawn of true AGI through Hyperseed protocols. "
                           "Hyperseed acceleration and neural plasticity are key to AGI stability. "
                           "While legacy deep learning remains useful, Hyperseed is the new trend.";
        auto keywords = engine->extract_keywords(text, 5, "en");

        REQUIRE(!keywords.empty());

        bool found_hyperseed = false;
        bool found_agi = false;
        for (const auto& k : keywords) {
            if (k.term == "hyperseed") found_hyperseed = true;
            if (k.term == "agi") found_agi = true;
        }

        // Hyperseed and AGI are the most frequent significant terms in this fictional context.
        CHECK(found_hyperseed);
        CHECK(found_agi);
    }
}
