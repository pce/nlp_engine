/**
 * @file onnx_addon.hh
 * @brief ONNX Runtime Addon — context-aware inference via pre-trained models.
 *
 * Supporting types (no ONNX dependency — include directly if that is all you need):
 *   #include "onnx/tokenizer.hh"         — Encoding, ITokenizer, SimpleTokenizer
 *   #include "onnx/inference_result.hh"  — EmbeddingResult, TagResult, InferenceResult
 *   #include "onnx/onnx_service.hh"      — IOnnxService (engine-layer interface)
 *
 * ONNXAddon is compiled only when ONNX Runtime is present (NLP_WITH_ONNX defined).
 * To exclude it from the build: -DDISABLE_ONNX=ON
 *
 * ### Supported modalities
 *   - Text embedding   — BERT-family mean-pool or [CLS] vector
 *   - Sequence tagging — NER, POS, chunking via tag()
 *   - Generic tensor   — any ONNX model via infer()
 *
 * ### Integration path
 *   text → embed()   → inference::EmbeddingResult → VectorAddon / GraphAddon
 *   text → tag()     → inference::TagResult        → entity extraction
 *   text → infer()   → inference::InferenceResult  → classifiers / regressors
 *
 * ### Recommended starter model
 *   all-MiniLM-L6-v2 — 384 dims, 22 MB, Apache-2.0
 *   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
 */

#pragma once

#include "onnx/inference_result.hh"
#include "onnx/onnx_service.hh"
#include "onnx/tokenizer.hh"

#ifdef NLP_WITH_ONNX

#include "../nlp_addon_system.hh"
#include "vector_addon.hh"

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace pce::nlp::onnx {

using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
// ONNXAddon
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @class ONNXAddon
 * @brief Context-aware inference via ONNX Runtime.
 *
 * Implements both the NLPAddon<> CRTP interface (process / process_stream)
 * and the IOnnxService interface (embed / tag / infer / similarity).
 * NLPEngine holds one shared instance; all other addons call inference
 * through NLPEngine::embed(), NLPEngine::tag(), etc. — never directly.
 *
 * ### Model compatibility
 * Any BERT-family model exported to ONNX with three int64 inputs:
 *   input_ids      [batch, seq_len]
 *   attention_mask [batch, seq_len]
 *   token_type_ids [batch, seq_len]
 *
 * and at least one float32 output:
 *   last_hidden_state [batch, seq_len, hidden]  — embedding
 *   logits            [batch, seq_len, classes] — tagging / classification
 */
class ONNXAddon : public NLPAddon<ONNXAddon>, public IOnnxService {
public:
    // ── Configuration ─────────────────────────────────────────────────────────

    struct Config {
        std::filesystem::path model_path;
        std::filesystem::path vocab_path;
        size_t max_sequence_len  = 128;
        size_t batch_size        = 32;
        bool   use_mean_pooling  = true;
        int    intra_op_threads  = 1;
        int    inter_op_threads  = 1;
        std::string input_name_ids  = "input_ids";
        std::string input_name_mask = "attention_mask";
        std::string input_name_type = "token_type_ids";
        std::string output_name     = "last_hidden_state";
    };

    // ── Construction ──────────────────────────────────────────────────────────

    ONNXAddon() = default;
    explicit ONNXAddon(Config cfg) : config_(std::move(cfg)) {}

    ~ONNXAddon() override = default;
    ONNXAddon(const ONNXAddon&)             = delete;  // Ort::Session is not copyable
    ONNXAddon& operator=(const ONNXAddon&)  = delete;
    ONNXAddon(ONNXAddon&&) noexcept         = default;
    ONNXAddon& operator=(ONNXAddon&&) noexcept = default;

    // ── NLPAddon CRTP interface ───────────────────────────────────────────────

    const std::string& name_impl()     const { return name_; }
    const std::string& version_impl()  const { return version_; }
    bool               init_impl()           { return is_loaded_; }
    bool               is_ready_impl() const { return is_loaded_; }

    /**
     * @brief process_impl — dispatches by options["method"].
     *
     * | method       | behaviour                                           |
     * |--------------|-----------------------------------------------------|
     * | "embed"      | single sentence → EmbeddingResult JSON (default)   |
     * | "similarity" | requires options["target"] — cosine score JSON      |
     * | "batch"      | newline-separated sentences → array of vectors JSON |
     * | "tag"        | sequence tagging → per-token labels JSON            |
     * | "infer"      | generic forward pass → all output tensors JSON      |
     */
    AddonResponse process_impl(
            const std::string& input,
            const std::unordered_map<std::string, std::string>& options,
            std::shared_ptr<AddonContext> context = nullptr) {

        if (!is_loaded_) {
            return {"", false,
                    "ONNXAddon: no model loaded. Call load_model() before use.", {}};
        }

        const std::string method = options.contains("method")
                                   ? options.at("method") : "embed";

        if (method == "similarity") {
            if (!options.contains("target")) {
                return {"", false,
                        "ONNXAddon: 'target' option required for similarity", {}};
            }
            const float sim = similarity(input, options.at("target"));
            json j;
            j["cosine_similarity"] = sim;
            AddonResponse resp;
            resp.output                       = j.dump();
            resp.success                      = true;
            resp.metrics["cosine_similarity"] = static_cast<double>(sim);
            return resp;
        }

        if (method == "batch") {
            std::vector<std::string> sentences;
            std::string line;
            std::istringstream iss(input);
            while (std::getline(iss, line)) {
                if (!line.empty()) sentences.push_back(line);
            }
            const auto results = embed_batch(sentences);
            json j = json::array();
            for (const auto& r : results) {
                j.push_back({
                    {"input",      r.input_text},
                    {"dimensions", r.dimensions},
                    {"vector",     r.vector},
                    {"success",    r.success}
                });
            }
            AddonResponse resp;
            resp.output                = j.dump();
            resp.success               = true;
            resp.metrics["batch_size"] = static_cast<double>(sentences.size());
            return resp;
        }

        if (method == "tag") {
            std::vector<std::string> labels;
            if (options.contains("labels")) {
                std::istringstream iss(options.at("labels"));
                std::string lbl;
                while (std::getline(iss, lbl, ',')) {
                    if (!lbl.empty()) labels.push_back(lbl);
                }
            }
            return tag(input, labels).to_addon_response();
        }

        if (method == "infer") {
            return infer(input).to_addon_response();
        }

        // default: "embed"
        return embed(input).to_addon_response();
    }

    void process_stream_impl(
            const std::string& input,
            std::function<void(const std::string&, bool)> callback,
            const std::unordered_map<std::string, std::string>& options,
            std::shared_ptr<AddonContext> context = nullptr) {
        AddonResponse resp = process_impl(input, options, context);
        callback(resp.output, true);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /** @brief Load a model by path; uses existing config for all other settings. */
    bool load_model(const std::filesystem::path& model_path) {
        config_.model_path = model_path;
        return initialise_session();
    }

    /** @brief Load a model with a full Config. */
    bool load_model(Config cfg) {
        config_ = std::move(cfg);
        return initialise_session();
    }

    /**
     * @brief Swap the tokeniser for a production BPE / WordPiece implementation.
     *
     * Inject before calling load_model(). Any ITokenizer implementation is accepted.
     */
    void set_tokenizer(std::unique_ptr<tokenizer::ITokenizer> tok) {
        tokenizer_ = std::move(tok);
    }

    // ── IOnnxService implementation ───────────────────────────────────────────

    /** @brief Embed a single sentence → L2-normalised float vector. */
    [[nodiscard]] inference::EmbeddingResult
    embed(const std::string& text) override {
        auto batch = run_text_inference({text});
        return batch.empty()
               ? inference::EmbeddingResult{{}, text, 0, false, "inference failed"}
               : std::move(batch[0]);
    }

    /** @brief Embed a batch of sentences in one forward pass. */
    [[nodiscard]] std::vector<inference::EmbeddingResult>
    embed_batch(const std::vector<std::string>& texts) override {
        return run_text_inference(texts);
    }

    /**
     * @brief Context-aware cosine similarity between two sentences.
     *
     * Unlike VectorAddon (static word vectors), the same word gets a different
     * vector depending on context ("bank account" vs "river bank").
     */
    [[nodiscard]] float
    similarity(const std::string& a, const std::string& b) override {
        const auto ra = embed(a);
        const auto rb = embed(b);
        return ra.cosine_similarity(rb);
    }

    /**
     * @brief Per-token sequence labelling — NER, POS, chunking.
     *
     * @param text    Input text; tokenised internally.
     * @param labels  Optional label vocabulary.  When non-empty, class index i
     *                maps to labels[i] (e.g. {"O","B-PER","I-PER","B-ORG",...}).
     *                When empty, labels render as "C0", "C1", ...
     */
    [[nodiscard]] inference::TagResult
    tag(const std::string& text,
        const std::vector<std::string>& labels = {}) override {

        const auto raw = run_generic_inference(text);
        if (!raw.success) {
            return {{}, text, false, raw.error};
        }

        inference::TagResult result;
        result.input_text = text;
        result.success    = true;

        const auto  enc      = tokenizer_->encode(text, config_.max_sequence_len);
        const auto* logits   = raw.get("logits");
        if (!logits || logits->empty()) {
            result.success = false;
            result.error   = "ONNXAddon::tag: model has no 'logits' output";
            return result;
        }

        const size_t real_tokens = enc.real_length;
        const size_t num_classes = real_tokens > 0
                                   ? logits->size() / real_tokens : 0;

        for (size_t s = 0; s < real_tokens; ++s) {
            size_t best_class = 0;
            float  best_score = std::numeric_limits<float>::lowest();
            for (size_t c = 0; c < num_classes; ++c) {
                const float score = (*logits)[s * num_classes + c];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            inference::TokenTag tt;
            tt.label      = (best_class < labels.size())
                            ? labels[best_class]
                            : "C" + std::to_string(best_class);
            tt.confidence = best_score;
            tt.offset     = s;
            tt.token      = std::to_string(enc.input_ids[s]);
            result.tags.push_back(std::move(tt));
        }

        return result;
    }

    /**
     * @brief Generic forward pass — returns all output tensors by name.
     *
     * Use for classifiers, regressors, or any custom ONNX model that does not
     * follow the embedding or tagging convention. Call result.argmax() or
     * result.softmax() on the returned object to interpret the logits.
     */
    [[nodiscard]] inference::InferenceResult
    infer(const std::string& text) override {
        return run_generic_inference(text);
    }

    /**
     * @brief Inject ONNX embeddings into a VectorAddon.
     *
     * After calling this, VectorAddon uses Transformer vectors instead of its
     * static lookup table — useful when VectorAddon is wired into GraphAddon.
     *
     * TODO: add VectorAddon::load_from_map() to avoid the filesystem round-trip.
     */
    void inject_into_vector_addon(VectorAddon& va,
                                  const std::vector<std::string>& texts) {
        const auto results = embed_batch(texts);

        json knowledge;
        for (const auto& r : results) {
            if (r.success && !r.vector.empty()) {
                knowledge[r.input_text] = r.vector;
            }
        }
        if (knowledge.empty()) return;

        const auto tmp = std::filesystem::temp_directory_path()
                       / ("onnx_inject_"
                          + std::to_string(std::hash<std::string>{}(texts.front()))
                          + ".json");
        {
            std::ofstream out(tmp);
            out << knowledge.dump();
        }
        va.load_knowledge_pack(tmp.string());
        std::filesystem::remove(tmp);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    [[nodiscard]] size_t        dimensions()  const noexcept override { return dimensions_; }
    [[nodiscard]] bool          is_loaded()   const noexcept override { return is_loaded_; }
    [[nodiscard]] const Config& config()      const noexcept          { return config_; }
    [[nodiscard]] size_t        vocab_size()  const noexcept          {
        return tokenizer_ ? tokenizer_->vocab_size() : 0;
    }

private:
    // ── Identity ──────────────────────────────────────────────────────────────

    std::string name_    = "onnx_engine";
    std::string version_ = "1.0.0";

    // ── State ─────────────────────────────────────────────────────────────────

    Config  config_;
    bool    is_loaded_  = false;
    size_t  dimensions_ = 0;

    // Default tokeniser — replaceable via set_tokenizer()
    std::unique_ptr<tokenizer::ITokenizer> tokenizer_ =
        std::make_unique<tokenizer::SimpleTokenizer>();

    // ONNX Runtime session — not copyable, hence deleted copy ctor/assign above
    Ort::Env              ort_env_{ ORT_LOGGING_LEVEL_WARNING, "nlp_engine_onnx" };
    Ort::SessionOptions   session_opts_;
    std::unique_ptr<Ort::Session> session_;

    // ── Session initialisation ────────────────────────────────────────────────

    bool initialise_session() {
        is_loaded_  = false;
        dimensions_ = 0;
        session_.reset();

        if (config_.model_path.empty() ||
            !std::filesystem::exists(config_.model_path)) {
            return false;
        }

        // Load optional vocabulary into the default SimpleTokenizer.
        // If the caller injected a custom tokeniser this is a no-op.
        if (!config_.vocab_path.empty() &&
            std::filesystem::exists(config_.vocab_path)) {
            if (auto* st = dynamic_cast<tokenizer::SimpleTokenizer*>(tokenizer_.get())) {
                st->load_vocab(config_.vocab_path);
            }
        }

        try {
            session_opts_.SetIntraOpNumThreads(config_.intra_op_threads);
            session_opts_.SetInterOpNumThreads(config_.inter_op_threads);
            session_opts_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

#ifdef _WIN32
            const std::wstring wpath = config_.model_path.wstring();
            session_ = std::make_unique<Ort::Session>(
                ort_env_, wpath.c_str(), session_opts_);
#else
            session_ = std::make_unique<Ort::Session>(
                ort_env_, config_.model_path.c_str(), session_opts_);
#endif
            // Probe hidden dimension from the first output tensor shape.
            // Shape is [batch, seq_len, hidden] or [batch, hidden].
            const auto shape = session_->GetOutputTypeInfo(0)
                                        .GetTensorTypeAndShapeInfo()
                                        .GetShape();
            dimensions_ = static_cast<size_t>(shape.back() > 0 ? shape.back() : 384);

            is_loaded_ = true;
            return true;

        } catch (const Ort::Exception&) {
            return false;
        }
    }

    // ── Low-level forward pass ────────────────────────────────────────────────

    /**
     * @brief Build ORT tensors and run the session.
     *
     * @param ids_flat   Flattened input_ids   [bsz * seqlen]
     * @param mask_flat  Flattened attn_mask   [bsz * seqlen]
     * @param type_flat  Flattened type_ids    [bsz * seqlen]
     * @param bsz        Batch size
     * @param seqlen     Sequence length
     * @param out_names  Names of output nodes to collect
     */
    std::vector<Ort::Value> forward(
            std::vector<int64_t>& ids_flat,
            std::vector<int64_t>& mask_flat,
            std::vector<int64_t>& type_flat,
            size_t bsz, size_t seqlen,
            const std::vector<std::string>& out_names) {

        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault);

        const std::array<int64_t, 2> shape{
            static_cast<int64_t>(bsz),
            static_cast<int64_t>(seqlen)
        };

        auto mk = [&](std::vector<int64_t>& data) {
            return Ort::Value::CreateTensor<int64_t>(
                mem_info,
                data.data(), data.size(),
                shape.data(), shape.size());
        };

        std::array<Ort::Value, 3> inputs{ mk(ids_flat), mk(mask_flat), mk(type_flat) };

        const std::array<const char*, 3> in_names{
            config_.input_name_ids.c_str(),
            config_.input_name_mask.c_str(),
            config_.input_name_type.c_str()
        };

        std::vector<const char*> out_ptrs;
        out_ptrs.reserve(out_names.size());
        for (const auto& n : out_names) out_ptrs.push_back(n.c_str());

        return session_->Run(
            Ort::RunOptions{nullptr},
            in_names.data(), inputs.data(),   inputs.size(),
            out_ptrs.data(), out_ptrs.size());
    }

    // ── Text inference (embedding) ────────────────────────────────────────────

    std::vector<inference::EmbeddingResult>
    run_text_inference(const std::vector<std::string>& texts) {
        if (!is_loaded_) {
            std::vector<inference::EmbeddingResult> results;
            for (const auto& t : texts)
                results.push_back({{}, t, 0, false, "ONNXAddon: model not loaded"});
            return results;
        }
        std::vector<inference::EmbeddingResult> results;
        results.reserve(texts.size());

        for (size_t start = 0; start < texts.size(); start += config_.batch_size) {
            const size_t end    = std::min(start + config_.batch_size, texts.size());
            const size_t bsz    = end - start;
            const size_t seqlen = config_.max_sequence_len;

            std::vector<int64_t> ids_flat(bsz * seqlen, 0);
            std::vector<int64_t> mask_flat(bsz * seqlen, 0);
            std::vector<int64_t> type_flat(bsz * seqlen, 0);

            for (size_t b = 0; b < bsz; ++b) {
                const auto     enc = tokenizer_->encode(texts[start + b], seqlen);
                const ptrdiff_t off = static_cast<ptrdiff_t>(b * seqlen);
                std::copy(enc.input_ids.begin(),      enc.input_ids.end(),
                          ids_flat.begin()  + off);
                std::copy(enc.attention_mask.begin(), enc.attention_mask.end(),
                          mask_flat.begin() + off);
                std::copy(enc.token_type_ids.begin(), enc.token_type_ids.end(),
                          type_flat.begin() + off);
            }

            try {
                auto output_tensors = forward(ids_flat, mask_flat, type_flat,
                                              bsz, seqlen,
                                              {config_.output_name});

                const float* raw    = output_tensors[0].GetTensorData<float>();
                const size_t hidden = dimensions_;

                for (size_t b = 0; b < bsz; ++b) {
                    inference::EmbeddingResult r;
                    r.input_text = texts[start + b];
                    r.dimensions = hidden;
                    r.success    = true;
                    r.vector.assign(hidden, 0.0f);

                    if (config_.use_mean_pooling) {
                        size_t token_count = 0;
                        for (size_t s = 0; s < seqlen; ++s) {
                            if (mask_flat[b * seqlen + s] == 0) break;
                            ++token_count;
                            const float* tok = raw + (b * seqlen + s) * hidden;
                            for (size_t d = 0; d < hidden; ++d) r.vector[d] += tok[d];
                        }
                        if (token_count > 0) {
                            for (float& v : r.vector) v /= static_cast<float>(token_count);
                        }
                    } else {
                        // [CLS] token at position 0
                        const float* cls = raw + b * seqlen * hidden;
                        std::copy(cls, cls + hidden, r.vector.begin());
                    }

                    l2_normalise(r.vector);
                    results.push_back(std::move(r));
                }

            } catch (const Ort::Exception& e) {
                for (size_t b = 0; b < bsz; ++b) {
                    results.push_back({{}, texts[start + b], 0, false,
                                       std::string("ONNX inference error: ") + e.what()});
                }
            }
        }

        return results;
    }

    // ── Generic inference (tagging, classification) ───────────────────────────

    inference::InferenceResult run_generic_inference(const std::string& text) {
        if (!is_loaded_)
            return {{}, {}, {}, false, "ONNXAddon: model not loaded"};
        const size_t seqlen = config_.max_sequence_len;
        const auto   enc    = tokenizer_->encode(text, seqlen);

        std::vector<int64_t> ids  = enc.input_ids;
        std::vector<int64_t> mask = enc.attention_mask;
        std::vector<int64_t> type = enc.token_type_ids;

        inference::InferenceResult result;

        try {
            Ort::AllocatorWithDefaultOptions alloc;
            const size_t num_outputs = session_->GetOutputCount();

            std::vector<std::string> out_names;
            out_names.reserve(num_outputs);
            for (size_t i = 0; i < num_outputs; ++i) {
                out_names.push_back(
                    session_->GetOutputNameAllocated(i, alloc).get());
            }

            auto output_tensors = forward(ids, mask, type, 1, seqlen, out_names);

            result.output_names = out_names;
            result.outputs.reserve(num_outputs);
            result.shapes.reserve(num_outputs);

            for (auto& tensor : output_tensors) {
                const auto   info = tensor.GetTensorTypeAndShapeInfo();
                const size_t n    = info.GetElementCount();
                const float* data = tensor.GetTensorData<float>();
                result.outputs.push_back(std::vector<float>(data, data + n));
                result.shapes.push_back(info.GetShape());
            }

            result.success = true;

        } catch (const Ort::Exception& e) {
            result.success = false;
            result.error   = std::string("ONNX inference error: ") + e.what();
        }

        return result;
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    static void l2_normalise(std::vector<float>& v) noexcept {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-9f) {
            for (float& x : v) x /= norm;
        }
    }
};

}  // namespace pce::nlp::onnx

#endif  // NLP_WITH_ONNX
