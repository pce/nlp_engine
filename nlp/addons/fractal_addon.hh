#ifndef FRACTAL_ADDON_HH
#define FRACTAL_ADDON_HH

#include "../nlp_addon_system.hh"
#include "markov_addon.hh"
#include "vector_addon.hh"
#include <cmath>
#include <sstream>
#include <random>
#include <algorithm>
#include <iterator>

namespace pce::nlp {

/**
 * @class FractalAddon
 * @brief Experimental text generator that uses recursive branching patterns.
 *
 * This addon creates "Fractal Text" by recursively splitting the generation
 * into branches, using a Markov source for local texture and an optional
 * Vector engine to maintain thematic consistency across branches.
 */
class FractalAddon : public INLPAddon {
public:
    FractalAddon() : name_("fractal_generator"), version_("1.0.0-experimental"), depth_(3), segment_length_(20), ready_(false) {
        std::random_device rd;
        gen_.seed(rd());
    }

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

    void set_markov_source(std::shared_ptr<MarkovAddon> source) {
        markov_source_ = source;
        if (source) ready_ = true;
    }

    void set_vector_engine(std::shared_ptr<VectorAddon> engine) {
        vector_engine_ = engine;
    }

    /**
     * @brief Process a fractal generation request.
     * @param options { "depth": "3", "length": "20", "temperature": "1.0" }
     */
    AddonResponse process(const std::string& input,
                         const std::unordered_map<std::string, std::string>& options,
                         std::shared_ptr<AddonContext> context = nullptr) override {

        if (!markov_source_) {
            return {"", false, "Fractal engine requires a Markov source instance.", {}};
        }

        int depth = options.count("depth") ? std::stoi(options.at("depth")) : 3;
        int seg_len = options.count("length") ? std::stoi(options.at("length")) : 20;

        // Ensure parameters are sane
        depth = std::clamp(depth, 0, 5); // Prevent stack overflow
        seg_len = std::clamp(seg_len, 5, 100);

        std::cout << "[Fractal] Starting generation. Depth: " << depth << " SegLen: " << seg_len << std::endl;

        std::string result = generate_recursive(input, depth, seg_len, options, context);

        AddonResponse resp;
        resp.output = result;
        resp.success = true;
        resp.metrics["depth"] = static_cast<double>(depth);
        resp.metrics["total_length"] = static_cast<double>(result.length());

        return resp;
    }

    bool is_ready() const override {
        return true;
    }

private:
    /**
     * @brief Recursive core of the fractal generator.
     */
    std::string generate_recursive(const std::string& seed, int depth, int segment_len,
                                 const std::unordered_map<std::string, std::string>& options,
                                 std::shared_ptr<AddonContext> context) {
        if (depth <= 0) {
            // Base case: Generate a standard Markov segment
            auto local_options = options;
            local_options["length"] = std::to_string(segment_len);

            auto resp = markov_source_->process(seed, local_options, context);
            return resp.output;
        }

        // --- Fractal Branching (Binary Split) ---

        // Branch A: Primary generation from current seed
        std::string branch_a = generate_recursive(seed, depth - 1, segment_len, options, context);

        // Extract context for Branch B
        // We use the last 2 words of Branch A to seed Branch B
        std::string bridge_seed = extract_context(branch_a, 2);

        // Branch B: Recursive variation
        std::string branch_b = generate_recursive(bridge_seed, depth - 1, segment_len, options, context);

        // Compose the segments with structural markers
        std::stringstream ss;
        ss << branch_a;
        ss << "\n\n[:: Branch Depth " << depth << " ::]\n\n";
        ss << branch_b;

        return ss.str();
    }

    /**
     * @brief Extracts the last N words from a text segment to act as a bridge seed.
     */
    std::string extract_context(const std::string& text, int word_count) {
        std::istringstream iss(text);
        std::vector<std::string> words((std::istream_iterator<std::string>(iss)),
                                        std::istream_iterator<std::string>());

        if (words.empty()) return "";

        int start = std::max(0, static_cast<int>(words.size()) - word_count);
        std::string res;
        for (size_t i = start; i < words.size(); ++i) {
            res += words[i];
            if (i < words.size() - 1) res += " ";
        }
        return res;
    }

    std::shared_ptr<MarkovAddon> markov_source_;
    std::shared_ptr<VectorAddon> vector_engine_;
    std::string name_;
    std::string version_;
    std::mt19937 gen_;
    int depth_;
    int segment_length_;
    bool ready_;
};

} // namespace pce::nlp

#endif // FRACTAL_ADDON_HH
