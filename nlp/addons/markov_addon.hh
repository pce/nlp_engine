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
#include <cmath>
#include <nlohmann/json.hpp>
#include "vector_addon.hh"

namespace pce::nlp {

using json = nlohmann::json;

/**
 * @class MarkovAddon
 * @brief High-performance C++23 Markov Chain Text Generator with N-Gram support.
 *
 * This version supports:
 * 1. Variable N-Grams (Bigrams, Trigrams, etc.)
 * 2. Softmax Temperature Sampling for creativity control.
 * 3. Nucleus (Top-P) Sampling.
 * 4. Hybrid Semantic Filtering via VectorAddon.
 */
class MarkovAddon : public NLPAddon<MarkovAddon>, public ITrainable {
private:
    /**
     * @brief Internal state for the Markov model.
     * Maps an N-Gram sequence (joined by space) to a map of next words and their frequencies.
     */
    std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> chain_;

    // Configurable N-Gram size (default to 2 for bigrams, 3 for trigrams)
    size_t n_gram_size_ = 2;

    // Optional semantic validator
    std::shared_ptr<VectorAddon> vector_engine_;

    std::string name_ = "markov_generator";
    std::string version_ = "2.0.0";
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
        // Keep alphanumeric for tokens
        s.erase(std::remove_if(s.begin(), s.end(),
                               [](unsigned char c){ return std::ispunct(c) && c != '\'' && c != '-'; }),
                s.end());
        return s;
    }

    /**
     * @brief Joins a window of words into a single key string.
     */
    std::string join_window(const std::vector<std::string>& window) const {
        std::string result;
        for (size_t i = 0; i < window.size(); ++i) {
            result += window[i];
            if (i < window.size() - 1) result += " ";
        }
        return result;
    }

public:
    MarkovAddon() = default;

    // --- NLPAddon Implementation ---

    const std::string& name_impl() const { return name_; }
    const std::string& version_impl() const { return version_; }

    void set_name(const std::string& new_name) { name_ = new_name; }

    /**
     * @brief Set the N-Gram context size. 2 = Bigram, 3 = Trigram.
     */
    void set_ngram_size(size_t n) { n_gram_size_ = n; }

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
     * @param options { "length": "50", "temperature": "1.0", "top_p": "0.9", "use_hybrid": "true" }
     */
    void process_stream_impl(const std::string& input,
                            std::function<void(const std::string& chunk, bool is_final)> callback,
                            const std::unordered_map<std::string, std::string>& options,
                            std::shared_ptr<AddonContext> context = nullptr) {
        if (!ready_) {
            callback("Error: Markov model not loaded", true);
            return;
        }

        int max_length = options.count("length") ? std::stoi(options.at("length")) : 50;
        float temperature = options.count("temperature") ? std::stof(options.at("temperature")) : 1.0f;
        float top_p = options.count("top_p") ? std::stof(options.at("top_p")) : 0.9f;
        bool use_hybrid = options.count("use_hybrid") ? (options.at("use_hybrid") == "true") : false;
        float semantic_threshold = options.count("semantic_filter") ? std::stof(options.at("semantic_filter")) : 0.3f;

        // Ensure n_gram_size is synced from options if provided
        if (options.count("n_gram")) {
            n_gram_size_ = std::max((size_t)2, (size_t)std::stoul(options.at("n_gram")));
        }

        std::vector<std::string> window;
        std::istringstream iss(input);
        std::string w;

        while (iss >> w) {
            std::string cleaned = clean_word(w);
            if (!cleaned.empty()) {
                window.push_back(cleaned);
            }
        }

        // Handle empty or too small seed
        if (window.empty()) {
            if (chain_.empty()) {
                callback("Error: Chain is empty. Please train the model first.", true);
                return;
            }
            auto it = chain_.begin();
            std::advance(it, std::uniform_int_distribution<size_t>(0, chain_.size() - 1)(gen_));
            std::istringstream ss(it->first);
            while (ss >> w) window.push_back(w);
        }

        // Ensure we don't exceed the required history for the N-Gram key
        if (window.size() >= n_gram_size_) {
            window.erase(window.begin(), window.begin() + (window.size() - (n_gram_size_ - 1)));
        }

        // Emit initial seed tokens to the stream
        for (const auto& word : window) callback(word + " ", false);

        for (int i = 0; i < max_length; ++i) {
            std::string key = join_window(window);
            auto it = chain_.find(key);

            // Backoff strategy: if trigram not found, try bigram, etc.
            while (it == chain_.end() && !window.empty()) {
                window.erase(window.begin());
                key = join_window(window);
                it = chain_.find(key);
            }

            if (it == chain_.end()) {
                // Total dead end: Jump to random start
                auto rand_it = chain_.begin();
                std::advance(rand_it, std::uniform_int_distribution<size_t>(0, chain_.size() - 1)(gen_));
                std::istringstream ss(rand_it->first);
                window.clear();
                while (ss >> w) window.push_back(w);
                callback("... " + window.back() + " ", false);
                continue;
            }

            const auto& possibilities = it->second;
            std::vector<std::pair<std::string, float>> scored_candidates;

            // 1. Temperature-based Softmax Scoring
            float sum_exp = 0.0f;
            for (const auto& [word, freq] : possibilities) {
                float score = std::pow(static_cast<float>(freq), 1.0f / std::max(0.01f, temperature));
                scored_candidates.push_back({word, score});
                sum_exp += score;
            }

            // Normalize
            for (auto& cand : scored_candidates) cand.second /= sum_exp;

            // 2. Hybrid Semantic Filtering
            if (use_hybrid && vector_engine_ && !window.empty()) {
                std::string context_word = window.back();
                for (auto& cand : scored_candidates) {
                    float sim = vector_engine_->calculate_similarity(context_word, cand.first);
                    if (sim < semantic_threshold) cand.second *= 0.1f; // Penalty
                }
            }

            // 3. Sort for Nucleus Sampling
            std::sort(scored_candidates.begin(), scored_candidates.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });

            // 4. Top-P (Nucleus) Filter
            float cumulative = 0.0f;
            std::vector<std::pair<std::string, float>> nucleus;
            for (const auto& cand : scored_candidates) {
                nucleus.push_back(cand);
                cumulative += cand.second;
                if (cumulative >= top_p) break;
            }

            // 5. Random Sample from Nucleus
            std::uniform_real_distribution<float> dist(0.0f, cumulative);
            float target = dist(gen_);
            float current_sum = 0.0f;
            std::string next_word;

            for (const auto& cand : nucleus) {
                current_sum += cand.second;
                if (current_sum >= target) {
                    next_word = cand.first;
                    break;
                }
            }

            if (next_word.empty()) next_word = nucleus.front().first;

            callback(next_word + " ", false);

            // Advance window
            window.push_back(next_word);
            if (window.size() >= n_gram_size_) {
                window.erase(window.begin());
            }
        }

        callback("", true);
    }

    AddonResponse process_impl(const std::string& input,
                              const std::unordered_map<std::string, std::string>& options,
                              std::shared_ptr<AddonContext> context = nullptr) {
        std::string result;
        process_stream_impl(input, [&](const std::string& chunk, bool is_final) {
            if (!is_final) result += chunk;
        }, options, context);

        AddonResponse resp;
        resp.output = result;
        resp.success = true;
        return resp;
    }

    bool is_ready() const override { return ready_; }

    /**
     * @brief Loads a Knowledge Pack from JSON.
     */
    bool load_knowledge_pack(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        json data;
        try { file >> data; } catch (...) { return false; }

        chain_.clear();
        if (data.contains("ngram_size")) n_gram_size_ = data["ngram_size"];

        auto model_data = data.contains("data") ? data["data"] : data;
        for (auto it = model_data.begin(); it != model_data.end(); ++it) {
            std::string key = it.key();
            for (auto next_it = it.value().begin(); next_it != it.value().end(); ++next_it) {
                chain_[key][next_it.key()] = next_it.value();
            }
        }

        ready_ = !chain_.empty();
        return ready_;
    }

    /**
     * @brief Trains a model using N-Grams.
     */
    bool train(const std::string& source_path, const std::string& model_output_path) override {
        std::ifstream file(source_path);
        if (!file.is_open()) return false;

        std::string word;
        std::vector<std::string> history;
        std::unordered_map<std::string, std::unordered_map<std::string, uint32_t>> temp_chain;

        while (file >> word) {
            word = clean_word(word);
            if (word.empty()) continue;

            if (history.size() == n_gram_size_ - 1) {
                std::string key = join_window(history);
                temp_chain[key][word]++;
            }

            history.push_back(word);
            if (history.size() >= n_gram_size_) {
                history.erase(history.begin());
            }
        }

        json output;
        output["ngram_size"] = n_gram_size_;
        output["data"] = temp_chain;
        output["metadata"] = {
            {"version", version_},
            {"engine", "pce_nlp_markov_v2"}
        };

        std::ofstream out_file(model_output_path);
        out_file << output.dump(2);
        return true;
    }

    float get_training_progress() const override { return ready_ ? 1.0f : 0.0f; }
};

} // namespace pce::nlp

#endif // MARKOV_ADDON_HH
