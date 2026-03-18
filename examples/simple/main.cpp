#include "nlp_engine.h"
#include <iostream>
#include <vector>
#include <string>

/**
 * Simple Example for pce::nlp::NLPEngine
 *
 * This example demonstrates:
 * 1. Language Detection
 * 2. Spell Checking
 * 3. Part-of-Speech Tagging
 * 4. Readability Analysis
 */

using namespace pce::nlp;

void print_separator(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

int main() {
    NLPEngine engine;

    // Example Text (German)
    std::string text_de = "Das ist ein beispiel für die automatische Textanalyse. Wir prüfen die Rechtschreibung.";

    // 1. Language Detection
    print_separator("Language Detection");
    auto lang = engine.detect_language(text_de);
    std::cout << "Detected Language: " << lang.language << " (Confidence: " << lang.confidence << ")\n";

    // 2. Spell Checking (with language override)
    print_separator("Spell Checking (German)");
    // "beispiel" is correct, "Rechtschreibung" is correct. Let's try a typo:
    std::string typo_text = "Wir lernen Deutsch in der Schulle.";
    auto corrections = engine.spell_check(typo_text, "de");

    if (corrections.empty()) {
        std::cout << "No spelling errors found.\n";
    } else {
        for (const auto& c : corrections) {
            std::cout << "Error: '" << c.original << "' -> Suggestion: '" << c.suggested << "'\n";
        }
    }

    // 3. POS Tagging & Stemming
    print_separator("POS Tagging & Stemming (English)");
    std::string text_en = "The quick brown foxes are jumping over the lazy dog.";
    auto tokens = engine.tokenize(text_en);
    auto tagged = engine.pos_tag(tokens, "en");

    std::cout << "Word [Tag] -> Stem\n";
    for (const auto& pair : tagged) {
        std::string stem = engine.stem(pair.first, "en");
        std::cout << pair.first << " [" << pair.second << "] -> " << stem << "\n";
    }

    // 4. Readability Analysis
    print_separator("Readability Analysis");
    auto metrics = engine.analyze_readability(text_en);
    std::cout << "Word Count: " << metrics.word_count << "\n";
    std::cout << "Sentence Count: " << metrics.sentence_count << "\n";
    std::cout << "Flesch-Kincaid Grade: " << metrics.flesch_kincaid_grade << "\n";
    std::cout << "Complexity: " << metrics.complexity << "\n";

    // 5. Entity Extraction
    print_separator("Entity Extraction");
    std::string contact_info = "Please send an email to support@example.com or visit https://pce-tools.io";
    auto entities = engine.extract_entities(contact_info);
    for (const auto& e : entities) {
        std::cout << "Found " << e.type << ": " << e.text << "\n";
    }

    return 0;
}
