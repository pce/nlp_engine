#ifndef DEDUPE_ADDON_HH
#define DEDUPE_ADDON_HH

#include "../nlp_addon_system.hh"
#include "vector_addon.hh"
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <regex>

namespace pce::nlp {

/**
 * @class DeduplicationAddon
 * @brief Advanced deduplication for granular pattern detection.
 * Supports detection and removal of repeated segments based on normalization rules.
 */
class DeduplicationAddon : public INLPAddon {
public:
    DeduplicationAddon() : name_("deduplication"), version_("2.2.0"), ready_(true) {}

    const std::string& name() const override { return name_; }
    const std::string& version() const override { return version_; }

    bool initialize() override { return true; }

    void process_stream(const std::string& input,
                        std::function<void(const std::string& chunk, bool is_final)> callback,
                        const std::unordered_map<std::string, std::string>& options,
                        std::shared_ptr<AddonContext> context = nullptr) override {
        auto resp = process(input, options, context);
        callback(resp.output, true);
    }

    void set_vector_engine(std::shared_ptr<VectorAddon> engine) {
        vector_engine_ = engine;
    }

    /**
     * @brief Process text by segmenting it into phrases/sentences and identifying duplicates.
     *
     * Options:
     * - mode: "detect" | "remove"
     * - min_length: minimum character length of a segment to be considered for deduplication
     * - skip_words: comma-separated list of words to ignore during normalization
     * - ignore_quotes: boolean string ("true"/"false") to strip quotes during comparison
     * - ignore_punctuation: boolean string ("true"/"false") to strip punctuation during comparison
     */
    AddonResponse process(const std::string& input,
                         const std::unordered_map<std::string, std::string>& options,
                         std::shared_ptr<AddonContext> context = nullptr) override {

        std::string mode = options.count("mode") ? options.at("mode") : "detect";
        size_t min_len_threshold = options.count("min_length") ? std::stoul(options.at("min_length")) : 1;
        bool ignore_quotes = options.count("ignore_quotes") && options.at("ignore_quotes") == "true";
        bool ignore_punctuation = options.count("ignore_punctuation") && options.at("ignore_punctuation") == "true";

        std::unordered_set<std::string> skip_set;
        if (options.count("skip_words") && !options.at("skip_words").empty()) {
            std::stringstream ss(options.at("skip_words"));
            std::string w;
            while (std::getline(ss, w, ',')) {
                if (!w.empty()) skip_set.insert(normalize_word(w));
            }
        }

        struct Segment {
            std::string raw;        // Original text including trailing punctuation/space
            std::string signature;  // Normalized version used for comparison
            size_t offset;
            size_t length;
            bool is_duplicate = false;
        };

        std::vector<Segment> segments;
        // Regex to split by sentence-ending punctuation while keeping the punctuation
        std::regex segment_regex(R"([^.!?\s][^.!?]*[.!?]*)");
        auto seg_begin = std::sregex_iterator(input.begin(), input.end(), segment_regex);
        auto seg_end = std::sregex_iterator();

        size_t last_pos = 0;
        for (std::sregex_iterator i = seg_begin; i != seg_end; ++i) {
            std::smatch match = *i;
            std::string raw = match.str();

            // Check for leading whitespace that might have been skipped by the regex
            if (match.position() > last_pos) {
                // If we are in remove mode, we might want to preserve the leading space
                // but for segmentation we usually attach it to the next segment or keep it.
            }

            std::string sig = create_signature(raw, skip_set, ignore_quotes, ignore_punctuation);

            segments.push_back({raw, sig, static_cast<size_t>(match.position()), raw.length(), false});
            last_pos = match.position() + raw.length();
        }

        std::unordered_set<std::string> seen_signatures;
        int dup_count = 0;

        for (auto& seg : segments) {
            if (seg.signature.empty()) continue;

            // Apply min_length check on the signature or the raw text?
            // Unit tests suggest min_length applies to the segment being compared.
            if (seg.signature.length() < min_len_threshold) continue;

            if (seen_signatures.count(seg.signature)) {
                seg.is_duplicate = true;
                dup_count++;
            } else {
                seen_signatures.insert(seg.signature);
            }
        }

        AddonResponse resp;
        resp.success = true;

        if (mode == "remove") {
            std::string result;
            bool first = true;
            for (const auto& seg : segments) {
                if (!seg.is_duplicate) {
                    if (!first && !result.empty() && result.back() != ' ' && seg.raw.front() != ' ') {
                        result += " ";
                    }
                    result += seg.raw;
                    first = false;
                }
            }
            // Trim trailing space if added
            if (!result.empty() && result.back() == ' ') result.pop_back();
            resp.output = result;
        } else {
            resp.output = input;
        }

        // Export duplicates as structured metadata
        int meta_idx = 0;
        for (const auto& seg : segments) {
            if (seg.is_duplicate) {
                std::string idx_str = std::to_string(meta_idx++);
                resp.metadata["dup_" + idx_str + "_text"] = seg.raw;
                resp.metadata["dup_" + idx_str + "_offset"] = std::to_string(seg.offset);
                resp.metadata["dup_" + idx_str + "_length"] = std::to_string(seg.length);
            }
        }

        resp.metrics["duplicates_found"] = static_cast<double>(dup_count);
        resp.metrics["has_duplicates"] = dup_count > 0 ? 1.0 : 0.0;

        return resp;
    }

    bool is_ready() const override { return ready_; }

private:
    std::string normalize_word(const std::string& s) {
        std::string res;
        for (unsigned char c : s) {
            if (!std::ispunct(c) && !std::isspace(c)) {
                res += static_cast<char>(std::tolower(c));
            }
        }
        return res;
    }

    std::string create_signature(const std::string& s,
                                const std::unordered_set<std::string>& skip_set,
                                bool ignore_quotes,
                                bool ignore_punctuation) {
        std::stringstream ss;
        std::string word;
        std::string input_copy = s;

        // Simple tokenizer for signature creation
        std::regex word_regex(R"(\S+)");
        auto words_begin = std::sregex_iterator(input_copy.begin(), input_copy.end(), word_regex);
        auto words_end = std::sregex_iterator();

        bool first = true;
        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::string w = i->str();

            if (ignore_quotes) {
                w.erase(std::remove(w.begin(), w.end(), '\"'), w.end());
                w.erase(std::remove(w.begin(), w.end(), '\''), w.end());
            }
            if (ignore_punctuation) {
                w.erase(std::remove_if(w.begin(), w.end(), ::ispunct), w.end());
            }

            std::string norm = normalize_word(w);
            if (norm.empty() || skip_set.count(norm)) continue;

            if (!first) ss << " ";
            ss << norm;
            first = false;
        }
        return ss.str();
    }

    std::shared_ptr<VectorAddon> vector_engine_;
    std::string name_;
    std::string version_;
    bool ready_;
};

} // namespace pce::nlp

#endif
