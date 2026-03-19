/**
 * @file main.cpp
 * @brief End-to-end example of training and loading a Markov Knowledge Pack.
 *
 * This example demonstrates:
 * 1. Initializing a MarkovAddon for training.
 * 2. Processing a raw text source into a Knowledge Pack (JSON).
 * 3. Loading that pack into an AsyncNLPEngine.
 * 4. Generating text from the trained model.
 */

#include "../../nlp/nlp_engine.hh"
#include "../../nlp/nlp_engine_async.hh"
#include "../../nlp/addons/markov_addon.hh"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>

namespace fs = std::filesystem;
using namespace pce::nlp;

int main(int argc, char* argv[]) {
    // 1. Setup paths
    std::string category = (argc > 1) ? argv[1] : "generic_novel";
    std::string source_path = "data/training/" + category + "_source.txt";
    std::string model_path = "data/models/" + category + ".json";

    std::cout << "--- NLP Engine Training Example ---" << std::endl;

    // Ensure directories exist
    fs::create_directories("data/models");

    if (!fs::exists(source_path)) {
        std::cerr << "[!] Source file not found: " << source_path << std::endl;
        std::cerr << "[!] Please run: python3 scripts/fetch_training_data.py --category novel" << std::endl;

        // Create a tiny dummy file for the sake of the example if it doesn't exist
        std::cout << "[*] Creating dummy training data for demonstration..." << std::endl;
        fs::create_directories("data/training");
        std::ofstream dummy(source_path);
        dummy << "The architect designed a building. The building was tall and modern. "
              << "The modern building stood in the center of the city.";
        dummy.close();
    }

    // 2. Offline Training Phase
    // We use the MarkovAddon directly as an ITrainable
    std::cout << "[*] Training '" << category << "' model..." << std::endl;

    auto trainer = std::make_shared<MarkovAddon>();
    if (trainer->train(source_path, model_path)) {
        std::cout << "[+] Training complete. Model saved to: " << model_path << std::endl;
    } else {
        std::cerr << "[-] Training failed!" << std::endl;
        return 1;
    }

    // 3. Runtime Loading Phase
    // Load the linguistic model (dictionaries, etc.)
    auto model = std::make_shared<NLPModel>();
    if (!model->load_from("data")) {
        std::cout << "[!] Warning: Standard linguistic data not found, continuing with empty model." << std::endl;
    }

    // Initialize the Async Engine
    AsyncNLPEngine engine(model);
    if (!engine.initialize()) {
        std::cerr << "[-] Failed to initialize AsyncNLPEngine" << std::endl;
        return 1;
    }

    // Register the trained Addon
    auto generator = std::make_shared<MarkovAddon>();
    if (generator->load_knowledge_pack(model_path)) {
        engine.add_addon(generator);
        std::cout << "[+] Addon '" << generator->name() << "' registered and ready." << std::endl;
    } else {
        std::cerr << "[-] Failed to load Knowledge Pack." << std::endl;
        return 1;
    }

    // 4. Execution Phase
    std::cout << "\n--- Text Generation ---" << std::endl;
    std::string seed = "The";
    std::unordered_map<std::string, std::string> options = {
        {"length", "20"},
        {"temperature", "0.8"}
    };

    std::cout << "Seed: \"" << seed << "\"" << std::endl;

    // Synchronous call to the addon
    std::string result = engine.process_sync(seed, "markov_generator", options);

    std::cout << "Generated output: " << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << result << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    engine.shutdown();
    return 0;
}
