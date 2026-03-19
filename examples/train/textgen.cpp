/**
 * @file textgen.cpp
 * @brief Advanced Markov generation example with benchmarking and hybrid options.
 *
 * This example demonstrates:
 * 1. Loading a pre-trained Markov Knowledge Pack.
 * 2. Benchmarking generation speed (Native vs. Hybrid Vector Mode).
 * 3. Using the streaming interface for incremental output.
 */

#include "../../nlp/nlp_engine_async.hh"
#include "../../nlp/addons/markov_addon.hh"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <memory>
#include <vector>

using namespace pce::nlp;

void print_stats(const std::string& label, double duration_ms, size_t word_count) {
    double wps = (word_count / (duration_ms / 1000.0));
    std::cout << "\n--- " << label << " ---" << std::endl;
    std::cout << "Duration:  " << std::fixed << std::setprecision(2) << duration_ms << " ms" << std::endl;
    std::cout << "Words:     " << word_count << std::endl;
    std::cout << "Speed:     " << std::fixed << std::setprecision(2) << wps << " words/sec" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string model_name = (argc > 1) ? argv[1] : "generic_novel";
    std::string model_path = "data/models/" + model_name + ".json";

    std::cout << ">>> NLP Engine Text Generation Benchmark <<<" << std::endl;

    // 1. Initialize Engine
    auto model = std::make_shared<NLPModel>();
    model->load_from("data");

    AsyncNLPEngine engine(model);
    engine.initialize();

    // 2. Load and Register Addon
    auto markov = std::make_shared<MarkovAddon>();
    if (!markov->load_knowledge_pack(model_path)) {
        std::cerr << "[!] Failed to load: " << model_path << "\nRun nlp_example_train first." << std::endl;
        return 1;
    }
    engine.add_addon(markov);

    std::string seed = "The";
    int length = 50;

    // --- SCENARIO A: FAST NATIVE MARKOV ---
    {
        std::unordered_map<std::string, std::string> options = {
            {"length", std::to_string(length)},
            {"use_hybrid", "false"},
            {"top_p", "0.9"}
        };

        auto start = std::chrono::high_resolution_clock::now();
        std::string result = engine.process_sync(seed, "markov_generator", options);
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        print_stats("Native Markov (O1)", duration, length);
        std::cout << "Preview: " << result.substr(0, 100) << "..." << std::endl;
    }

    // --- SCENARIO B: HYBRID VECTOR SEMANTICS ---
    {
        std::unordered_map<std::string, std::string> options = {
            {"length", std::to_string(length)},
            {"use_hybrid", "true"},
            {"top_p", "0.8"},
            {"semantic_filter", "0.3"}
        };

        std::cout << "\n[*] Running Hybrid Vector mode (Calculating similarities...)" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        std::string result = engine.process_sync(seed, "markov_generator", options);
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        print_stats("Hybrid Vector Mode", duration, length);
        std::cout << "Preview: " << result.substr(0, 100) << "..." << std::endl;
    }

    // --- SCENARIO C: STREAMING INTERFACE ---
    {
        std::cout << "\n--- Streaming Output (Native) ---" << std::endl;
        std::unordered_map<std::string, std::string> options = {{"length", "30"}};

        auto callback = [](const std::string& chunk, bool is_final) {
            std::cout << chunk << std::flush;
            if (is_final) std::cout << "\n[Stream End]" << std::endl;
        };

        engine.stream_text(seed, model_name, callback, options);

        // Wait a moment for the async stream to finish if using internal threads
        // In this simple example, stream_text blocks the caller for Markov
        // but it shows how the callback pattern works.
    }

    engine.shutdown();
    return 0;
}
