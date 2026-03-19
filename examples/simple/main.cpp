#include "nlp_engine.hh"
#include <iostream>
#include <vector>
#include <string>
#include <memory>

/**
 * Simple Example for pce::nlp::NLPEngine (Decoupled Architecture)
 *
 * This example demonstrates:
 * 1. Explicit Model Loading (NLPModel)
 * 2. Processing with NLPEngine
 * 3. Language-aware features (German/English)
 */

using namespace pce::nlp;

void print_separator(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}

int main() {
    // 1. Initialize the Data Model
    // The model holds all the heavy dictionaries and lexicons.
    auto model = std::make_shared<NLPModel>();

    std::cout << "Loading NLP resources from 'data/' directory...\n";
    if (!model->load_from("data")) {
        std::cerr << "Error: Could not load NLP resources. Ensure 'data/' directory exists with required files.\n";
        return 1;
    }

    // 2. Initialize the Engine with the Model
    // The engine is now stateless and refers to the model for data.
    NLPEngine engine(model);

    // --- Example 1: German Analysis ---
    std::string text_de = "Das ist ein tolles Beispiel für die automatische Analyse. Wir lernen Deutsch.";

    print_separator("German Analysis");
    auto lang = engine.detect_language(text_de);
    std::cout << "Detected Language: " << lang.language << " (Confidence: " << lang.confidence << ")\n";

    auto sentiment = engine.analyze_sentiment(text_de, "de");
    std::cout << "Sentiment: " << sentiment.label << " (Score: " << sentiment.score << ")\n";

    // --- Example 2: Spell Checking ---
    print_separator("Spell Checking");
    std::string typo_text = "I am very hapy today."; // 'hapy' instead of 'happy'
    auto corrections = engine.spell_check(typo_text, "en");

    if (corrections.empty()) {
        std::cout << "No spelling errors found.\n";
    } else {
        for (const auto& c : corrections) {
            std::cout << "Error: '" << c.original << "' -> Suggested: '" << c.suggested << "'\n";
        }
    }

    // --- Example 3: Readability & ICALL Features ---
    print_separator("Readability & Linguistics (English)");
    std::string text_en = "The quick brown fox jumps over the lazy dog. Linguistic analysis is fascinating.";

    auto readability = engine.analyze_readability(text_en);
    std::cout << "Complexity: " << readability.complexity << "\n";
    std::cout << "Flesch-Kincaid Grade: " << readability.flesch_kincaid_grade << "\n";

    auto tokens = engine.tokenize(text_en);
    auto tagged = engine.pos_tag(tokens, "en");

    std::cout << "\nFirst 5 Tokens [POS Tag]:\n";
    for (size_t i = 0; i < std::min(size_t(5), tagged.size()); ++i) {
        std::cout << "  " << tagged[i].first << " [" << tagged[i].second << "]\n";
    }

    // --- Example 4: Safety & Ethics ---
    print_separator("Toxicity Detection");
    std::string toxic_text = "You are a complete idiot and I hate you.";
    auto toxicity = engine.detect_toxicity(toxic_text, "en");

    if (toxicity.is_toxic) {
        std::cout << "WARNING: Toxic content detected!\n";
        std::cout << "Category: " << toxicity.category << "\n";
        std::cout << "Triggers: ";
        for (const auto& t : toxicity.triggers) std::cout << t << " ";
        std::cout << "\n";
    } else {
        std::cout << "Content is clean.\n";
    }

    return 0;
}
