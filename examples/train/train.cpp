/**
 * @file train.cpp
 * @brief Professional Offline Training CLI for Markov Knowledge Packs.
 *
 * This utility handles the "Training Phase" of the NLP Engine lifecycle.
 * It consumes raw text corpora and transforms them into serialized
 * N-Gram probability models (Knowledge Packs) that can be loaded
 * by the C++ native engine or the FastAPI bridge.
 *
 * Lifecycle:
 * 1. Load Raw Text -> 2. Tokenize -> 3. Build Probability Map -> 4. Serialize to JSON
 */

#include "../../nlp/nlp_engine.hh"
#include "../../nlp/nlp_engine_async.hh"
#include "../../nlp/addons/markov_addon.hh"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <chrono>

namespace fs = std::filesystem;
using namespace pce::nlp;

/**
 * @brief Main training entry point
 * @param argc 2
 * @param argv [category_name] (e.g., 'technical_docs', 'poetry')
 */
int main(int argc, char* argv[]) {
    // --- 1. Path Configuration ---
    std::string category = (argc > 1) ? argv[1] : "generic_novel";
    std::string source_path = "data/training/" + category + "_source.txt";
    std::string model_path = "data/models/" + category + ".json";

    std::cout << "==========================================" << std::endl;
    std::cout << "   NLP ENGINE: OFFLINE TRAINING PHASE     " << std::endl;
    std::cout << "==========================================" << std::endl;

    // Ensure directory structure exists
    try {
        fs::create_directories("data/models");
        fs::create_directories("data/training");
    } catch (const fs::filesystem_error& e) {
        std::cerr << "[Critical] Directory creation failed: " << e.what() << std::endl;
        return 1;
    }

    // --- 2. Data Validation ---
    if (!fs::exists(source_path)) {
        std::cout << "[Info] Source file not found: " << source_path << std::endl;
        std::cout << "[*] Generating synthetic training corpus for demonstration..." << std::endl;

        std::ofstream dummy(source_path);
        dummy << "The engineer designed a system. The system was robust and scalable. "
              << "The scalable system utilized C++ for high performance. "
              << "High performance is critical for linguistic processing at scale. "
              << "The processing engine handles millions of tokens per second.";
        dummy.close();

        std::cout << "[+] Created: " << source_path << " (" << fs::file_size(source_path) << " bytes)" << std::endl;
    }

    // --- 3. Training Execution ---
    // We instantiate the MarkovAddon directly as it implements ITrainable
    auto trainer = std::make_shared<MarkovAddon>();

    std::cout << "[*] Starting Tokenization & Model Construction..." << std::endl;
    std::cout << "    Target Model: " << model_path << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    if (trainer->train(source_path, model_path)) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "[✓] Training Successful!" << std::endl;
        std::cout << "    Processing Time: " << duration.count() << "ms" << std::endl;
        std::cout << "    Model Size:      " << fs::file_size(model_path) / 1024.0 << " KB" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Next Step: Run 'nlp_example_textgen " << category << "' to test generation." << std::endl;
    } else {
        std::cerr << "[✗] Training failed! Check if source file is accessible and non-empty." << std::endl;
        return 1;
    }

    // --- 4. Optional Validation ---
    // Verify the pack is loadable
    if (trainer->load_knowledge_pack(model_path)) {
        std::cout << "[Info] Integrity check passed: Model is loadable." << std::endl;
    }

    return 0;
}
