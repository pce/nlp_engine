#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include "../nlp/addons/markov_addon.hh"

namespace fs = std::filesystem;
using namespace pce::nlp;

/**
 * @file nlp_trainer.cpp
 * @brief Simple CLI Trainer for NLP Engine Addons.
 *
 * Usage: ./nlp_trainer --type markov --source input.txt --output model.json
 */

/**
 * @brief Prints help information for the trainer CLI.
 */
void print_usage() {
    std::cout << "NLP Engine Trainer CLI\n";
    std::cout << "Usage: nlp_trainer [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --type <addon_type>    Type of addon to train (e.g., markov)\n";
    std::cout << "  --source <path>        Path to the source training text file\n";
    std::cout << "  --output <path>        Path where the Knowledge Pack (JSON) will be saved\n";
    std::cout << "  --help                 Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string type;
    std::string source;
    std::string output;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--type" && i + 1 < argc) type = argv[++i];
        else if (arg == "--source" && i + 1 < argc) source = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output = argv[++i];
        else if (arg == "--help") {
            print_usage();
            return 0;
        }
    }

    if (type.empty() || source.empty() || output.empty()) {
        std::cerr << "Error: Missing required arguments.\n";
        print_usage();
        return 1;
    }

    if (!fs::exists(source)) {
        std::cerr << "Error: Source file not found: " << source << "\n";
        return 1;
    }

    std::cout << "[Trainer] Initializing training for type: " << type << "\n";
    std::cout << "[Trainer] Source: " << source << "\n";
    std::cout << "[Trainer] Target: " << output << "\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    if (type == "markov") {
        MarkovAddon trainer;
        std::cout << "[Trainer] Starting Markov chain extraction...\n";

        if (trainer.train(source, output)) {
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end_time - start_time;

            std::cout << "[Trainer] Success! Knowledge Pack generated in "
                      << std::fixed << std::setprecision(2) << diff.count() << "s\n";
            std::cout << "[Trainer] Model saved to: " << fs::absolute(output) << "\n";
        } else {
            std::cerr << "[Trainer] Failed to train Markov model.\n";
            return 1;
        }
    } else {
        std::cerr << "Error: Unsupported addon type '" << type << "'.\n";
        return 1;
    }

    return 0;
}
