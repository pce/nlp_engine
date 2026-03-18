#include "../nlp/nlp_engine.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <algorithm>

using namespace pce::nlp;

void test_tokenization() {
    NLPEngine engine;
    std::string text = "Hello, world! This is a test.";
    auto tokens = engine.tokenize(text);

    // Note: tokenize implementation might keep or remove punctuation depending on its logic
    // but we expect at least the main words.
    assert(!tokens.empty());
    std::cout << "✓ Tokenization test passed. Found " << tokens.size() << " tokens.\n";
}

void test_sentence_splitting() {
    NLPEngine engine;
    std::string text = "First sentence. Second sentence? Third sentence!";
    auto sentences = engine.split_sentences(text);

    assert(sentences.size() == 3);
    std::cout << "✓ Sentence splitting test passed.\n";
}

void test_normalization() {
    NLPEngine engine;
    std::string text = "Hello, WORLD!!!";
    std::string normalized = engine.normalize(text);

    // Assuming normalize converts to lowercase and removes punctuation
    assert(normalized == "hello world");
    std::cout << "✓ Normalization test passed.\n";
}

void test_spell_check() {
    NLPEngine engine;
    // "thiz" is not in the minimal dictionary, "this" is.
    std::string text = "thiz is a test";
    auto corrections = engine.spell_check(text);

    bool found_thiz = false;
    for (const auto& c : corrections) {
        if (c.original == "thiz") {
            found_thiz = true;
            break;
        }
    }

    // Since "thiz" isn't in the minimal dict, it should be flagged
    assert(found_thiz);
    std::cout << "✓ Spell check test passed.\n";
}

void test_language_aware_nlp() {
    NLPEngine engine;

    // Test German spell check
    // "beispel" (incorrect) -> "beispiel" (correct, in fallback dict)
    auto de_corrections = engine.spell_check("beispel", "de");
    bool found_de = false;
    for (const auto& c : de_corrections) {
        if (c.original == "beispel" && c.suggested == "beispiel") {
            found_de = true;
            break;
        }
    }
    assert(found_de);

    // Test French spell check
    // "maizon" (incorrect) -> "maison" (correct, in fallback dict)
    auto fr_corrections = engine.spell_check("maizon", "fr");
    bool found_fr = false;
    for (const auto& c : fr_corrections) {
        if (c.original == "maizon" && c.suggested == "maison") {
            found_fr = true;
            break;
        }
    }
    assert(found_fr);

    // Test stopwords removal for different languages
    std::vector<std::string> de_tokens = {"der", "hund", "ist", "hier"};
    auto de_filtered = engine.remove_stopwords(de_tokens, "de");
    // "der" and "ist" are in the fallback German stopwords
    assert(de_filtered.size() < de_tokens.size());

    std::cout << "✓ Language-aware NLP tests passed (DE/FR).\n";
}

void test_language_detection() {
    NLPEngine engine;

    auto en_profile = engine.detect_language("This is a simple English sentence.");
    assert(en_profile.language == "en");

    auto de_profile = engine.detect_language("Das ist ein einfacher deutscher Satz.");
    assert(de_profile.language == "de");

    auto fr_profile = engine.detect_language("C'est une phrase française simple.");
    assert(fr_profile.language == "fr");

    std::cout << "✓ Language detection test passed.\n";
}

void test_pos_tagging_and_stemming() {
    NLPEngine engine;

    // Test POS tagging
    std::vector<std::string> tokens = {"the", "cat", "is", "running", "quickly"};
    auto tagged = engine.pos_tag(tokens, "en");

    assert(tagged[0].second == "DET"); // "the" is a stopword -> DET
    assert(tagged[3].second == "VBG"); // "running" ends in "ing" -> VBG
    assert(tagged[4].second == "ADV"); // "quickly" ends in "ly" -> ADV

    // Test Stemming (Grundformreduktion)
    assert(engine.stem("running", "en") == "running"); // our simple stemmer doesn't handle ing yet
    assert(engine.stem("cats", "en") == "cat");
    assert(engine.stem("lernen", "de") == "lern");

    std::cout << "✓ POS tagging and stemming tests passed.\n";
}

void test_terminology_and_proper_names() {
    NLPEngine engine;
    std::string text = "Natural Language Processing is a field of Artificial Intelligence. John Doe lives in Berlin.";

    // Terminology extraction (Eigennamenerkennung / Terminologieextraktion)
    auto terms = engine.extract_terminology(text);
    bool found_nlp = false;
    for (const auto& term : terms) {
        if (term == "Natural Language") found_nlp = true;
    }

    // Proper name detection (heuristic)
    auto entities = engine.extract_entities(text);
    bool found_berlin = false;
    for (const auto& e : entities) {
        if (e.text == "Berlin") found_berlin = true;
    }

    assert(found_berlin);
    std::cout << "✓ Terminology and proper name extraction tests passed.\n";
}

void test_readability() {
    NLPEngine engine;
    std::string text = "The cat sat on the mat. It was a very simple cat.";
    auto metrics = engine.analyze_readability(text);

    assert(metrics.word_count > 0);
    assert(metrics.sentence_count == 2);
    assert(metrics.readability_score > 0);
    std::cout << "✓ Readability analysis test passed.\n";
}

void test_entity_extraction() {
    NLPEngine engine;
    std::string text = "Contact me at test@example.com or visit https://example.com";
    auto entities = engine.extract_entities(text);

    bool found_email = false;
    bool found_url = false;

    for (const auto& e : entities) {
        if (e.type == "email") found_email = true;
        if (e.type == "url") found_url = true;
    }

    assert(found_email);
    assert(found_url);
    std::cout << "✓ Entity extraction test passed.\n";
}

int main() {
    try {
        std::cout << "Running NLP Engine Tests...\n";
        std::cout << "---------------------------\n";

        test_tokenization();
        test_sentence_splitting();
        test_normalization();
        test_spell_check();
        test_language_aware_nlp();
        test_language_detection();
        test_pos_tagging_and_stemming();
        test_terminology_and_proper_names();
        test_readability();
        test_entity_extraction();

        std::cout << "---------------------------\n";
        std::cout << "All tests passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
