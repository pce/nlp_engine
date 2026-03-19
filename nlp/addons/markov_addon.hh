#ifndef MARKOV_ADDON_HH
#define MARKOV_ADDON_HH

#include "../nlp_addon_system.hh"
#include "../nlp_engine.hh"
#include <unordered_map>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "vector_addon.hh"

namespace pce::nlp {

using json = nlohmann::json;

/**
 * @class MarkovAddon
 * @brief High-performance C++23 Markov Chain Text Generator.
 *
 * Implements the NLPAddon CRTP interface. This addon is designed to be
 * read-only at runtime once a "Knowledge Pack" is loaded.
 */
class MarkovAddon : public NLPAddon<MarkovAddon>, public ITrainable {
private:
    // N-Gram support: Sequence of words -> { NextWord: Frequency }
    // Using a space-separated string as key for the n-gram sequence
    std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> chain_;

    // Cached total counts for faster probability normalization
    std::unordered_map<std::string, uint64_t> totals_;

    // Optional semantic validator
    std::shared_ptr<VectorAddon> vector_engine_;

    std::string name_ = "markov_generator";
    std::string version_ = "1.0.2";
    bool ready_ = false;

    // Random engine for generation
    mutable std::mt19937 gen_{std::random_device{}()};

    /**
     * @brief Normalizes a string to lowercase for consistent matching.
     */
    std::string to_lower(std::string s) const {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return s;
    }

    /**
     * @brief Cleans a word for lookup (lowercase + remove punctuation).
     */
    std::string clean_word(std::string s) const {
        s = to_lower(s);
        s.erase(std::remove_if(s.begin(), s.end(),
                               [](unsigned char c){ return std::ispunct(c); }),
                s.end());
        return s;
    }

public:
    MarkovAddon() = default;

    // --- NLPAddon Implementation ---

    const std::string& name_impl() const { return name_; }
    const std::string& version_impl() const { return version_; }

    void set_name(const std::string& new_name) { name_ = new_name; }

    /**
     * @brief Attach a vector engine for semantic rule-based post-processing.
     */
    void set_vector_engine(std::shared_ptr<VectorAddon> engine) {
        vector_engine_ = engine;
    }

    bool init_impl() {
        return true;
    }

    /**
     * @brief Process text generation based on a seed.
     * @param input The seed word or phrase.
     * @param options { "length": "50", "temperature": "1.0" }
     * @param context Optional persistent state for session tracking.
     */
    AddonResponse process_impl(const std::string& input,
                              const std::unordered_map<std::string, std::string>& options,
                              std::shared_ptr<AddonContext> context = nullptr) {
        if (!ready_) return {"", false, "Markov model not loaded", {}};

        int max_length = options.count("length") ? std::stoi(options.at("length")) : 50;
        float top_p = options.count("top_p") ? std::stof(options.at("top_p")) : 0.9f;
        float semantic_threshold = options.count("semantic_filter") ? std::stof(options.at("semantic_filter")) : 0.0f;

        // Use context history if available to influence seed selection
        std::string effective_seed = input;
        if (effective_seed.empty() && context && !context->history.empty()) {
            effective_seed = context->history.back();
        }

        std::string result = generate_text_advanced(effective_seed, max_length, top_p, semantic_threshold);

        // Update context history for follow-up calls
        if (context && !result.empty()) {
            context->history.push_back(result);
        }

        AddonResponse resp;
        resp.output = result;
        resp.success = true;
        resp.metrics["tokens_generated"] = static_cast<double>(max_length);
        return resp;
    }

    /**
     * @brief Checks if a Knowledge Pack is currently loaded.
     */
    bool is_ready() const override { return ready_; }

    // --- Model Management (Read-Only) ---

    /**
     * @brief Loads a pre-trained JSON Knowledge Pack.
     * @param path File path to the JSON model.
     */
    bool load_knowledge_pack(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        json data;
        try {
            file >> data;
        } catch (...) {
            return false;
        }

        chain_.clear();
        totals_.clear();

        for (auto it = data.begin(); it != data.end(); ++it) {
            std::string word = clean_word(it.key());
            if (word.empty()) continue;

            uint64_t word_total = 0;
            for (auto next_it = it.value().begin(); next_it != it.value().end(); ++next_it) {
                uint32_t freq = next_it.value();
                std::string next_word = clean_word(next_it.key());
                if (next_word.empty()) continue;

                chain_[word][next_word] = freq;
                word_total += freq;
            }
            if (word_total > 0) {
                totals_[word] = word_total;
            }
        }

        ready_ = !chain_.empty();
        return ready_;
    }

    // --- ITrainable Implementation ---

    /**
     * @brief Trains a Markov model from source text and exports to JSON.
     */
    bool train(const std::string& source_path, const std::string& model_output_path) override {
        std::ifstream file(source_path);
        if (!file.is_open()) return false;

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        std::istringstream iss(content);
        std::string prev_word, current_word;
        std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> temp_chain;

        while (iss >> current_word) {
            current_word = clean_word(current_word);
            if (current_word.empty()) continue;

            if (!prev_word.empty()) {
                temp_chain[prev_word][current_word]++;
            }
            prev_word = current_word;
        }

        if (temp_chain.empty()) return false;

        json output_data = temp_chain;
        std::ofstream out_file(model_output_path);
        if (!out_file.is_open()) return false;

        out_file << output_data.dump(2);
        return true;
    }

    float get_training_progress() const override { return ready_ ? 1.0f : 0.0f; }

private:
    /**
     * @brief Advanced text generation with Nucleus Sampling (Top-P) and Semantic Filtering.
     */
    std::string generate_text_advanced(const std::string& seed, int max_length, float top_p, float semantic_threshold) {
        if (chain_.empty()) return "";

        std::string current;
        std::stringstream ss;
        std::vector<std::string> window; // Slinding window for n-gram context

        // Initialize seed
        std::istringstream iss(seed);
        std::string w;
        while (iss >> w) {
            std::string cleaned = clean_word(w);
            if (!cleaned.empty()) {
                window.push_back(cleaned);
                if (ss.tellp() > 0) ss << " ";
                ss << cleaned;
            }
        }

        if (window.empty()) {
            auto it = chain_.begin();
            std::advance(it, std::uniform_int_distribution<size_t>(0, chain_.size() - 1)(gen_));
            current = it->first;
            ss << current;
            window.push_back(current);
        } else {
            current = window.back();
        }

        for (int i = 0; i < max_length; ++i) {
            auto it = chain_.find(current);
            if (it == chain_.end()) {
                // Dead end jump
                auto next_it = chain_.begin();
                std::advance(next_it, std::uniform_int_distribution<size_t>(0, chain_.size() - 1)(gen_));
                current = next_it->first;
                ss << ". " << current;
                window = {current};
                continue;
            }

            const auto& possibilities = it->second;
            std::vector<std::pair<std::string, uint32_t>> candidates(possibilities.begin(), possibilities.end());

            // Sort by frequency for Top-P sampling
            std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) {
                return a.second > b.second;
            });

            // Apply Semantic Filtering if Vector Engine is attached
            if (vector_engine_ && semantic_threshold > 0.0f) {
                auto filtered = candidates;
                candidates.clear();
                for (const auto& cand : filtered) {
                    float sim = vector_engine_->calculate_similarity(current, cand.first);
                    if (sim >= semantic_threshold) {
                        candidates.push_back(cand);
                    }
                }
                if (candidates.empty()) candidates = filtered; // Fallback if all filtered
            }

            // Nucleus Sampling (Top-P)
            uint64_t total = 0;
            for (const auto& c : candidates) total += c.second;

            uint64_t running_total = 0;
            uint64_t target_cutoff = static_cast<uint64_t>(total * top_p);
            std::vector<std::pair<std::string, uint32_t>> nucleus;

            for (const auto& c : candidates) {
                nucleus.push_back(c);
                running_total += c.second;
                if (running_total >= target_cutoff) break;
            }

            // Sample from nucleus
            std::uniform_int_distribution<uint64_t> dist(1, std::max((uint64_t)1, running_total));
            uint64_t target = dist(gen_);
            uint64_t cumulative = 0;
            std::string next_word;

            for (const auto& c : nucleus) {
                cumulative += c.second;
                if (cumulative >= target) {
                    next_word = c.first;
                    break;
                }
            }

            if (next_word.empty()) break;

            ss << " " << next_word;
            current = next_word;
        }

        return finalize_text(ss.str());
    }

    /**
     * @brief Internal text generation logic using weighted random sampling.
     * Handles multi-word seeds by taking the last known valid word.
     */
    std::string generate_text(const std::string& seed, int max_length) {
        return generate_text_advanced(seed, max_length, 1.0f, 0.0f);
    }

    /**
     * @brief Post-processing for punctuation and formatting.
     */
    std::string finalize_text(std::string res) {
        if (res.empty()) return "";

        // Capitalize first letter
        if (std::islower(static_cast<unsigned char>(res[0]))) {
            res[0] = std::toupper(static_cast<unsigned char>(res[0]));
        }

        // Balance quotes: remove leading/trailing lone quotes and ensure pairs
        size_t quote_count = std::count(res.begin(), res.end(), '"');
        if (quote_count % 2 != 0) {
            // If it starts with a quote but doesn't end with one, add one
            if (res.front() == '"') res += '"';
            // If it ends with a quote but doesn't start with one, add one
            else if (res.back() == '"') res = '"' + res;
            // Otherwise, just strip all quotes to be safe
            else res.erase(std::remove(res.begin(), res.end(), '"'), res.end());
        }

        // Ensure it ends with a sentence-ending punctuation
        if (!res.empty()) {
            char last = res.back();
            if (last != '.' && last != '!' && last != '?' && last != '"') {
                res += ".";
            } else if (last == '"' && res.size() > 1) {
                char prev = res[res.size() - 2];
                if (prev != '.' && prev != '!' && prev != '?') {
                    res.insert(res.size() - 1, ".");
                }
            }
        }

        return res;
    }
};

} // namespace pce::nlp

#endif // MARKOV_ADDON_HH
