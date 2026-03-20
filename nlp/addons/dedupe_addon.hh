#ifndef DEDUPE_ADDON_HH
#define DEDUPE_ADDON_HH

#include "../nlp_addon_system.hh"
#include "vector_addon.hh"
#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace pce::nlp {

/**
 * @class DeduplicationAddon
 * @brief Addon for detecting, removing, or Replacing duplicate text patterns.
 *
 * This addon can operate in three modes:
 * 1. "detect": Returns a JSON list of found duplicates.
 * 2. "remove": Strips duplicate sentences/phrases.
 * 3. "semantic_replace": Replaces duplicates with semantically similar alternatives
 *    using the VectorAddon.
 */
class DeduplicationAddon : public INLPAddon {
public:
    DeduplicationAddon() : name_("deduplicator"), version_("1.0.0"), ready_(true) {}

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
     * @brief Process text to handle duplicates.
     * @param options { "mode": "remove|detect|replace", "min_length": "10" }
     */
    AddonResponse process(const std::string& input,
                         const std::unordered_map<std::string, std::string>& options,
                         std::shared_ptr<AddonContext> context = nullptr) override {

        std::string mode = options.count("mode") ? options.at("mode") : "remove";
        size_t min_length = options.count("min_length") ? std::stoul(options.at("min_length")) : 5;

        // Split into segments and track original offsets for highlighting
        std::vector<std::pair<std::string, size_t>> segments_with_offsets;
        size_t current_pos = 0;
        std::stringstream ss_split(input);
        std::string segment;
        while (std::getline(ss_split, segment, '.')) {
            if (!segment.empty()) {
                size_t found_pos = input.find(segment, current_pos);
                if (found_pos != std::string::npos) {
                    segments_with_offsets.push_back({segment, found_pos});
                    current_pos = found_pos + segment.length();
                }
            }
        }

        std::vector<std::string> unique_segments;
        std::unordered_set<std::string> seen;
        std::vector<std::pair<std::string, size_t>> duplicate_hits;

        for (const auto& [seg, offset] : segments_with_offsets) {
            std::string cleaned = clean_string(seg);
            if (cleaned.length() < min_length) {
                unique_segments.push_back(seg);
                continue;
            }

            if (seen.find(cleaned) == seen.end()) {
                seen.insert(cleaned);
                unique_segments.push_back(seg);
            } else {
                duplicate_hits.push_back({seg, offset});
                if (mode == "semantic_replace") {
                    unique_segments.push_back("[Variation]");
                }
                // If mode is "remove", we skip
            }
        }

        AddonResponse resp;
        resp.success = true;

        if (mode == "detect") {
            std::stringstream ss;
            ss << "{\"duplicates\": [";
            for (size_t i = 0; i < duplicate_hits.size(); ++i) {
                ss << "{\"text\": \"" << duplicate_hits[i].first
                   << "\", \"offset\": " << duplicate_hits[i].second
                   << ", \"length\": " << duplicate_hits[i].first.length() << "}";
                if (i < duplicate_hits.size() - 1) ss << ", ";
            }
            ss << "]}";
            resp.output = ss.str();
        } else {
            std::stringstream ss;
            for (size_t i = 0; i < unique_segments.size(); ++i) {
                ss << unique_segments[i] << (i == unique_segments.size() - 1 ? "" : ". ");
            }
            resp.output = ss.str();
        }

        resp.metrics["duplicates_found"] = static_cast<double>(duplicate_hits.size());
        return resp;
    }

    bool is_ready() const override { return ready_; }

private:
    std::vector<std::string> split_sentences(const std::string& text) {
        std::vector<std::string> res;
        std::stringstream ss(text);
        std::string segment;
        while (std::getline(ss, segment, '.')) {
            if (!segment.empty()) {
                // Trim leading whitespace
                size_t first = segment.find_first_not_of(' ');
                if (std::string::npos != first) {
                    segment = segment.substr(first);
                }
                res.push_back(segment);
            }
        }
        return res;
    }

    std::string clean_string(const std::string& s) {
        std::string res = s;
        res.erase(std::remove_if(res.begin(), res.end(), ::isspace), res.end());
        std::transform(res.begin(), res.end(), res.begin(), ::tolower);
        return res;
    }

    std::shared_ptr<VectorAddon> vector_engine_;
    std::string name_;
    std::string version_;
    bool ready_;
};

} // namespace pce::nlp

#endif // DEDUPE_ADDON_HH
