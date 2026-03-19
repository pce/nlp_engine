#include "nlp_engine_async.hh"
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <algorithm>

namespace pce::nlp {

// ============ Helper: Simple ID Generator ============

static std::string generate_task_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::stringstream ss;
    ss << "task_" << std::hex << std::setw(16) << std::setfill('0') << dis(gen);
    return ss.str();
}

// ============ AsyncTaskManager Implementation ============

std::string AsyncTaskManager::submit_task(std::function<AsyncResult()> task) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    std::string task_id = generate_task_id();
    tasks_[task_id] = std::async(std::launch::async, [task]() {
        return task();
    });
    return task_id;
}

AsyncResult AsyncTaskManager::get_result(const std::string& task_id, bool wait) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    auto it = tasks_.find(task_id);
    if (it == tasks_.end()) {
        return {"", false, "Task not found", task_id};
    }

    if (!wait) {
        auto status = it->second.wait_for(std::chrono::seconds(0));
        if (status != std::future_status::ready) {
            return {"", false, "Task still running", task_id};
        }
    }

    try {
        AsyncResult res = it->second.get();
        tasks_.erase(it);
        return res;
    } catch (const std::exception& e) {
        return {"", false, e.what(), task_id};
    }
}

void AsyncTaskManager::cancel_task(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    tasks_.erase(task_id);
}

bool AsyncTaskManager::is_task_complete(const std::string& task_id) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    auto it = tasks_.find(task_id);
    if (it == tasks_.end()) return false;
    return it->second.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// ============ AsyncNLPEngine Implementation ============

AsyncNLPEngine::AsyncNLPEngine(std::shared_ptr<NLPModel> model)
    : model_(model), is_running_(false) {
    task_manager_ = std::make_unique<AsyncTaskManager>();
}

AsyncNLPEngine::~AsyncNLPEngine() {
    shutdown();
}

bool AsyncNLPEngine::initialize() {
    if (!model_ || !model_->is_ready()) {
        return false;
    }
    is_running_ = true;
    return true;
}

bool AsyncNLPEngine::shutdown() {
    is_running_ = false;
    return true;
}

bool AsyncNLPEngine::register_plugin(std::shared_ptr<AsyncNLPPlugin> plugin) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    plugins_["default"] = plugin;
    return true;
}

bool AsyncNLPEngine::unregister_plugin(const std::string& name) {
    std::lock_guard<std::mutex> lock(plugins_mutex_);
    return plugins_.erase(name) > 0;
}

std::string AsyncNLPEngine::process_sync(
    const std::string& text,
    const std::string& method,
    const std::unordered_map<std::string, std::string>& options
) {
    if (!is_running_ || !model_) return "{}";

    NLPEngine engine(model_);
    std::string lang = options.count("lang") ? options.at("lang") : "en";

    if (method == "language" || method == "detect_language") {
        auto res = engine.detect_language(text);
        return engine.language_to_json(res).dump();
    } else if (method == "sentiment" || method == "analyze_sentiment") {
        auto res = engine.analyze_sentiment(text, lang);
        return engine.sentiment_to_json(res).dump();
    } else if (method == "spell_check") {
        auto res = engine.spell_check(text, lang);
        return engine.corrections_to_json(res).dump();
    } else if (method == "readability" || method == "analyze_readability") {
        auto res = engine.analyze_readability(text);
        return engine.readability_to_json(res).dump();
    } else if (method == "terminology" || method == "extract_terminology") {
        auto res = engine.extract_terminology(text, lang);
        return json(res).dump();
    } else if (method == "keywords" || method == "extract_keywords") {
        auto res = engine.extract_keywords(text, 10, lang);
        return engine.keywords_to_json(res).dump();
    } else if (method == "tokenize") {
        auto res = engine.tokenize(text);
        return json(res).dump();
    }

    return "{\"error\": \"Unknown method: " + method + "\"}";
}

std::string AsyncNLPEngine::process_text_async(
    const std::string& text,
    const std::string& plugin_name,
    StreamCallback stream_callback,
    const std::unordered_map<std::string, std::string>& options
) {
    if (!is_running_) return "";

    return task_manager_->submit_task([this, text, stream_callback, options]() {
        NLPEngine engine(model_);

        if (stream_callback) {
            stream_callback("Starting batch analysis...\n", false);
        }

        auto lang = engine.detect_language(text);
        if (stream_callback) {
            stream_callback("Language: " + lang.language + "\n", false);
        }

        auto sentiment = engine.analyze_sentiment(text, lang.language);
        if (stream_callback) {
            stream_callback("Sentiment: " + sentiment.label + "\n", false);
        }

        if (stream_callback) {
            stream_callback("Analysis Complete.\n", true);
        }

        return AsyncResult{"Success", true, "", ""};
    });
}

std::string AsyncNLPEngine::submit_task(
    std::function<AsyncResult()> task,
    const std::string& task_name
) {
    return task_manager_->submit_task(task);
}

AsyncResult AsyncNLPEngine::get_task_result(const std::string& task_id) {
    return task_manager_->get_result(task_id);
}

void AsyncNLPEngine::stream_text(
    const std::string& text,
    const std::string& plugin_name,
    StreamCallback callback,
    const std::unordered_map<std::string, std::string>& options
) {
    if (!callback) return;

    // Use task_manager to run the streaming task.
    // This provides a stable lifetime for the task and prevents the detachment crash.
    task_manager_->submit_task([this, text, callback, options]() {
        callback("Initializing analysis stream...\n", false);

        if (!model_) {
            callback("[Error] Model pointer is null\n", true);
            return AsyncResult{"Error", false, "Model pointer null", ""};
        }

        NLPEngine engine(model_);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 1. Language Detection
        LanguageProfile lang{.language = "en", .confidence = 0.5f};

        try {
            // 1. Language Detection
            callback("[Log] Starting language detection...\n", false);
            lang = engine.detect_language(text);
            std::string lang_msg = "Language: " + lang.language + " • confidence: " + std::to_string((int)(lang.confidence * 100)) + "%\n";
            callback(lang_msg, false);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));

            // 2. POS Tagging
            bool do_pos = false;
            if (options.count("pos_tagging") && options.at("pos_tagging") == "true") {
                do_pos = true;
            }

            if (do_pos) {
                callback("[Log] Starting tokenization...\n", false);
                auto tokens = engine.tokenize(text);
                callback("[Log] Starting POS tagging for " + std::to_string(tokens.size()) + " tokens...\n", false);

                auto tags = engine.pos_tag(tokens, lang.language);
                callback("[Log] POS tagging complete.\n", false);

                std::string pos_header = "Linguistic Analysis (" + std::to_string(tags.size()) + " tokens):\n";
                callback(pos_header, false);

                // Group by POS for better visualization in the log
                std::string tag_cloud = "Tags: ";
                for(size_t i = 0; i < std::min(tags.size(), (size_t)15); ++i) {
                    tag_cloud += tags[i].first + "/" + tags[i].second + " ";
                }
                if (tags.size() > 15) tag_cloud += "...";
                callback(tag_cloud + "\n", false);
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            }

            // 3. Sentiment
            callback("[Log] Starting sentiment analysis...\n", false);
            auto sentiment = engine.analyze_sentiment(text, lang.language);
            std::string sent_msg = "Sentiment: " + sentiment.label + " • score: " + std::to_string(sentiment.score).substr(0, 5) + "\n";
            callback(sent_msg, false);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));

            // 4. Terminology Extraction (when requested via options)
            bool extract_terms = false;
            if (options.count("terminology") && options.at("terminology") == "true") {
                extract_terms = true;
            }

            if (extract_terms) {
                callback("[Log] Starting terminology extraction...\n", false);
                auto terms = engine.extract_terminology(text, lang.language);
                std::string term_msg = "Terminology: Found " + std::to_string(terms.size()) + " technical terms.\n";
                callback(term_msg, false);

                if (!terms.empty()) {
                    std::string term_preview = "Keywords: ";
                    for(size_t i = 0; i < std::min(terms.size(), (size_t)5); ++i) {
                        term_preview += terms[i] + (i < std::min(terms.size(), (size_t)5) - 1 ? ", " : "");
                    }
                    callback(term_preview + "\n", false);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            }

            // 5. Readability
            callback("[Log] Starting readability analysis...\n", false);
            auto metrics = engine.analyze_readability(text);
            std::string grade_str = std::isnan(metrics.flesch_kincaid_grade) ? "N/A" : std::to_string(metrics.flesch_kincaid_grade).substr(0, 4);
            std::string read_msg = "Complexity: " + metrics.complexity + " • Grade: " + grade_str + "\n";
            callback(read_msg, false);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));

            // 6. Toxicity
            callback("[Log] Starting toxicity detection...\n", false);
            auto toxicity = engine.detect_toxicity(text, lang.language);
            if (toxicity.is_toxic) {
                callback("Warning: Content flagged as " + toxicity.category + "\n", false);
            } else {
                callback("Content Safety: Clean\n", false);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            callback("Finished.\n", true);
        } catch (const std::exception& e) {
            callback("[Error] Linguistic core failure: " + std::string(e.what()) + "\n", true);
            return AsyncResult{"Error", false, e.what(), ""};
        } catch (...) {
            callback("[Error] Unknown crash in linguistic core\n", true);
            return AsyncResult{"Error", false, "Unknown crash", ""};
        }

        return AsyncResult{"Success", true, "", ""};
    });
}

} // namespace pce::nlp
