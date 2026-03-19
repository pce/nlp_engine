#include "nlp_engine_async.hh"
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <algorithm>
#include "addons/markov_addon.hh"

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
    // We allow initialization even if the model is missing,
    // as addons may provide their own linguistic logic.
    is_running_ = true;
    return true;
}

bool AsyncNLPEngine::shutdown() {
    is_running_ = false;
    return true;
}

bool AsyncNLPEngine::add_addon(std::shared_ptr<INLPAddon> addon) {
    if (!addon) return false;
    std::lock_guard<std::mutex> lock(addons_mutex_);
    addons_[addon->name()] = addon;
    return true;
}

bool AsyncNLPEngine::remove_addon(const std::string& name) {
    std::lock_guard<std::mutex> lock(addons_mutex_);
    return addons_.erase(name) > 0;
}

bool AsyncNLPEngine::has_addon(const std::string& name) {
    std::lock_guard<std::mutex> lock(addons_mutex_);
    return addons_.find(name) != addons_.end();
}

std::string AsyncNLPEngine::process_sync(
    const std::string& text,
    const std::string& method,
    const std::unordered_map<std::string, std::string>& options
) {
    if (!is_running_) {
        return "{\"error\": \"Engine not running. Call initialize() first.\"}";
    }

    // 1. Check for Addon (with detailed diagnostics)
    {
        std::lock_guard<std::mutex> lock(addons_mutex_);
        auto it = addons_.find(method);
        if (it != addons_.end()) {
            auto addon = it->second;
            if (!addon) {
                return "{\"error\": \"Addon pointer is null for method: " + method + "\"}";
            }

            if (!addon->is_ready()) {
                return "{\"error\": \"Addon '" + method + "' is registered but not ready (model not loaded?)\"}";
            }

            try {
                auto resp = addon->process(text, options);
                if (resp.success) {
                    if (resp.output.empty()) {
                        return "{\"status\": \"success\", \"output\": \"\", \"diagnostic\": \"Addon returned empty string but success=true\"}";
                    }
                    return resp.output;
                } else {
                    return "{\"error\": \"" + (resp.error_message.empty() ? "Unknown addon error" : resp.error_message) + "\"}";
                }
            } catch (const std::exception& e) {
                return "{\"error\": \"Exception during addon execution: " + std::string(e.what()) + "\"}";
            }
        }
    }

    // 2. Fallback to core engine methods
    if (!model_ || !model_->is_ready()) {
        return "{\"error\": \"Base linguistic model not loaded for core method: " + method + "\"}";
    }

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

    // Diagnostic info for unmatched method
    std::string registered_addons = "";
    {
        std::lock_guard<std::mutex> lock(addons_mutex_);
        for (auto const& [name, ptr] : addons_) {
            registered_addons += name + " ";
        }
    }

    return "{\"error\": \"Unknown method or addon: " + method + "\", \"registered_addons\": \"" + registered_addons + "\"}";
}

std::string AsyncNLPEngine::process_text_async(
    const std::string& text,
    const std::string& addon_name,
    StreamCallback stream_callback,
    const std::unordered_map<std::string, std::string>& options
) {
    if (!is_running_) return "";

    return task_manager_->submit_task([this, text, addon_name, stream_callback, options]() {
        // Check for Addon
        {
            std::lock_guard<std::mutex> lock(addons_mutex_);
            auto it = addons_.find(addon_name);
            if (it != addons_.end()) {
                if (stream_callback) stream_callback("Invoking addon: " + addon_name + "...\n", false);
                auto resp = it->second->process(text, options);
                if (stream_callback) stream_callback(resp.output, true);
                return AsyncResult{resp.output, resp.success, resp.error_message, ""};
            }
        }

        if (!model_) {
            if (stream_callback) stream_callback("[Error] Model not loaded\n", true);
            return AsyncResult{"Error", false, "Model not loaded", ""};
        }

        NLPEngine engine(model_);
        if (stream_callback) stream_callback("Starting batch analysis...\n", false);
        auto lang = engine.detect_language(text);
        if (stream_callback) stream_callback("Language: " + lang.language + "\n", false);
        auto sentiment = engine.analyze_sentiment(text, lang.language);
        if (stream_callback) stream_callback("Sentiment: " + sentiment.label + "\n", false);
        if (stream_callback) stream_callback("Analysis Complete.\n", true);

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
    const std::string& addon_name,
    StreamCallback callback,
    const std::unordered_map<std::string, std::string>& options
) {
    if (!callback) return;

    task_manager_->submit_task([this, text, addon_name, callback, options]() {
        callback("Initializing analysis stream...\n", false);

        // Check for Addon first
        {
            std::lock_guard<std::mutex> lock(addons_mutex_);
            auto it = addons_.find(addon_name);
            if (it != addons_.end()) {
                auto resp = it->second->process(text, options);
                callback(resp.output, true);
                return AsyncResult{resp.output, resp.success, resp.error_message, ""};
            }
        }

        if (!model_) {
            callback("[Error] Model pointer is null\n", true);
            return AsyncResult{"Error", false, "Model pointer null", ""};
        }

        NLPEngine engine(model_);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        try {
            callback("[Log] Starting language detection...\n", false);
            auto lang = engine.detect_language(text);
            callback("Language: " + lang.language + " • confidence: " + std::to_string((int)(lang.confidence * 100)) + "%\n", false);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));

            if (options.count("pos_tagging") && options.at("pos_tagging") == "true") {
                auto tokens = engine.tokenize(text);
                auto tags = engine.pos_tag(tokens, lang.language);
                std::string tag_cloud = "Tags: ";
                for(size_t i = 0; i < std::min(tags.size(), (size_t)15); ++i) {
                    tag_cloud += tags[i].first + "/" + tags[i].second + " ";
                }
                callback(tag_cloud + "\n", false);
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            }

            callback("[Log] Starting sentiment analysis...\n", false);
            auto sentiment = engine.analyze_sentiment(text, lang.language);
            callback("Sentiment: " + sentiment.label + " • score: " + std::to_string(sentiment.score).substr(0, 5) + "\n", false);
            std::this_thread::sleep_for(std::chrono::milliseconds(150));

            if (options.count("terminology") && options.at("terminology") == "true") {
                auto terms = engine.extract_terminology(text, lang.language);
                callback("Terminology: Found " + std::to_string(terms.size()) + " technical terms.\n", false);
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            }

            auto metrics = engine.analyze_readability(text);
            std::string grade_str = std::isnan(metrics.flesch_kincaid_grade) ? "N/A" : std::to_string(metrics.flesch_kincaid_grade).substr(0, 4);
            callback("Complexity: " + metrics.complexity + " • Grade: " + grade_str + "\n", false);

            callback("Finished.\n", true);
        } catch (const std::exception& e) {
            callback("[Error] Linguistic core failure: " + std::string(e.what()) + "\n", true);
            return AsyncResult{"Error", false, e.what(), ""};
        }

        return AsyncResult{"Success", true, "", ""};
    });
}

} // namespace pce::nlp
