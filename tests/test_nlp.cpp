#include "../nlp/nlp_engine.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

using namespace pce::nlp;

/**
 * Global setup for tests.
 * Loads the model once to be shared across test cases.
 */
std::shared_ptr<NLPModel> setup_model() {
    static std::shared_ptr<NLPModel> model = nullptr;
    if (!model) {
        model = std::make_shared<NLPModel>();
        // Note: In CMake environment, 'data/' is copied to the build folder
        if (!model->load_from("data")) {
            std::cerr << "Warning: Could not load data directory. Some tests may fail due to empty dictionaries.\n";
        }
    }
    return model;
}

void test_tokenization() {
    auto model = setup_model();
    NLPEngine engine(model);
    std::string text = "The quick brown fox.";
    auto tokens = engine.tokenize(text);
    assert(tokens.size() == 4);
    assert(tokens[0] == "the");
    std::cout << "✓ Tokenization test passed.\n";
}

void test_sentence_splitting() {
    auto model = setup_model();
    NLPEngine engine(model);
    std::string text = "First sentence. Second sentence? \"Is this third?\" Yes!";
    auto sentences = engine.split_sentences(text);
    assert(sentences.size() == 4);
    assert(sentences[0] == "First sentence.");
    std::cout << "✓ Sentence splitting test passed.\n";
}

void test_normalization() {
    auto model = setup_model();
    NLPEngine engine(model);
    std::string text = "Hello, World!";
    std::string normalized = engine.normalize(text);
    // normalization typically lowercases and removes punctuation in this engine
    assert(normalized == "hello world");
    std::cout << "✓ Normalization test passed.\n";
}

void test_spell_check() {
    auto model = setup_model();
    NLPEngine engine(model);
    if (!model->is_ready()) {
        std::cout << "⚠ Skipping spell check (no data).\n";
        return;
    }
    auto corrections = engine.spell_check("I am hapy", "en");
    bool found = false;
    for (const auto& c : corrections) {
        if (c.original == "hapy") found = true;
    }
    assert(found);
    std::cout << "✓ Spell check test passed.\n";
}

void test_language_detection() {
    auto model = setup_model();
    NLPEngine engine(model);
    auto en_profile = engine.detect_language("The quick brown fox jumps over the lazy dog.");
    assert(en_profile.language == "en");
    if (model->is_ready()) {
        auto de_profile = engine.detect_language("Das ist ein einfacher deutscher Satz.");
        assert(de_profile.language == "de");
    }
    std::cout << "✓ Language detection test passed.\n";
}

void test_pos_tagging_and_stemming() {
    auto model = setup_model();
    NLPEngine engine(model);
    // Stemming
    assert(engine.stem("running", "en") == "run");
    // POS Tagging
    auto tokens = engine.tokenize("The dog runs.");
    auto tags = engine.pos_tag(tokens, "en");
    assert(!tags.empty());
    std::cout << "✓ POS tagging and stemming test passed.\n";
}

void test_terminology_and_proper_names() {
    auto model = setup_model();
    NLPEngine engine(model);
    std::string text = "The CEO of Apple is Tim Cook.";
    auto terms = engine.extract_terminology(text, "en");
    assert(!terms.empty());
    std::cout << "✓ Terminology extraction test passed.\n";
}

void test_readability() {
    auto model = setup_model();
    NLPEngine engine(model);
    std::string complex = "Natural Language Processing is a subfield of linguistics and artificial intelligence.";
    auto metrics = engine.analyze_readability(complex);
    assert(metrics.word_count > 5);
    assert(metrics.flesch_kincaid_grade > 0);
    std::cout << "✓ Readability test passed.\n";
}

void test_entity_extraction() {
    auto model = setup_model();
    NLPEngine engine(model);
    std::string text = "Contact me at info@example.com or visit our office.";
    auto entities = engine.extract_entities(text, "en");
    bool found_email = false;
    for (const auto& e : entities) {
        if (e.type == "EMAIL") found_email = true;
    }
    assert(found_email);
    std::cout << "✓ Entity extraction test passed.\n";
}

void test_sentiment_and_toxicity() {
    auto model = setup_model();
    NLPEngine engine(model);
    if (!model->is_ready()) {
        std::cout << "⚠ Skipping sentiment/toxicity (no data).\n";
        return;
    }
    // Sentiment
    auto pos = engine.analyze_sentiment("This is a great day", "en");
    assert(pos.label == "positive");
    // Toxicity
    auto toxic = engine.detect_toxicity("You are a stupid idiot");
    assert(toxic.is_toxic == true);
    std::cout << "✓ Sentiment and Toxicity tests passed.\n";
}

int main() {
    try {
        std::cout << "Running NLPEngine Comprehensive Tests...\n";
        std::cout << "--------------------------------------\n";

        test_tokenization();
        test_sentence_splitting();
        test_normalization();
        test_spell_check();
        test_language_detection();
        test_pos_tagging_and_stemming();
        test_terminology_and_proper_names();
        test_readability();
        test_entity_extraction();
        test_sentiment_and_toxicity();

        std::cout << "--------------------------------------\n";
        std::cout << "All tests passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Tests failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
