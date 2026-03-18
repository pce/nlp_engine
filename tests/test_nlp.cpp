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
    auto model = std::make_shared<NLPModel>();
    // Note: In CMake environment, 'data/' is copied to the build folder
    if (!model->load_from("data")) {
        std::cerr << "Warning: Could not load data directory. Some tests may fail due to empty dictionaries.\n";
    }
    return model;
}

void test_architecture_decoupling() {
    auto model = setup_model();
    NLPEngine engine(model);

    std::string text = "This is a test.";
    auto tokens = engine.tokenize(text);
    assert(!tokens.empty());
    std::cout << "✓ Architecture decoupling test passed.\n";
}

void test_language_detection() {
    auto model = setup_model();
    NLPEngine engine(model);

    // English detection
    auto en_profile = engine.detect_language("The quick brown fox jumps over the lazy dog.");
    assert(en_profile.language == "en");

    // German detection (if data is loaded)
    auto de_profile = engine.detect_language("Das ist ein einfacher deutscher Satz.");
    if (model->is_ready()) {
        assert(de_profile.language == "de");
    }

    std::cout << "✓ Language detection test passed.\n";
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

void test_spell_check() {
    auto model = setup_model();
    NLPEngine engine(model);

    if (!model->is_ready()) {
        std::cout << "⚠ Skipping spell check test (no data).\n";
        return;
    }

    // 'hapy' should be flagged if 'happy' is in dictionary_en.txt
    auto corrections = engine.spell_check("I am hapy", "en");
    bool found = false;
    for (const auto& c : corrections) {
        if (c.original == "hapy") found = true;
    }
    assert(found);
    std::cout << "✓ Spell check test passed.\n";
}

void test_sentiment_analysis() {
    auto model = setup_model();
    NLPEngine engine(model);

    if (!model->is_ready()) {
        std::cout << "⚠ Skipping sentiment test (no data).\n";
        return;
    }

    auto pos = engine.analyze_sentiment("This is a great and happy day", "en");
    assert(pos.label == "positive");

    auto neg = engine.analyze_sentiment("This is a bad and sad day", "en");
    assert(neg.label == "negative");

    std::cout << "✓ Sentiment analysis test passed.\n";
}

void test_toxicity_detection() {
    auto model = setup_model();
    NLPEngine engine(model);

    if (!model->is_ready()) {
        std::cout << "⚠ Skipping toxicity test (no data).\n";
        return;
    }

    auto toxic = engine.detect_toxicity("You are a stupid idiot");
    assert(toxic.is_toxic == true);
    assert(!toxic.triggers.empty());

    auto clean = engine.detect_toxicity("Hello my friend, how are you?");
    assert(clean.is_toxic == false);

    std::cout << "✓ Toxicity detection test passed.\n";
}

void test_readability() {
    auto model = setup_model();
    NLPEngine engine(model);

    std::string complex = "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.";
    auto metrics = engine.analyze_readability(complex);

    assert(metrics.word_count > 10);
    assert(metrics.flesch_kincaid_grade > 0);
    std::cout << "✓ Readability test passed.\n";
}

int main() {
    try {
        std::cout << "Running NLPEngine Refactored Tests...\n";
        std::cout << "------------------------------------\n";

        test_architecture_decoupling();
        test_language_detection();
        test_sentence_splitting();
        test_spell_check();
        test_sentiment_analysis();
        test_toxicity_detection();
        test_readability();

        std::cout << "------------------------------------\n";
        std::cout << "All tests passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Tests failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
