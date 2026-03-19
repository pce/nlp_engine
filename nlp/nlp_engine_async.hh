#ifndef NLP_ENGINE_ASYNC_H
#define NLP_ENGINE_ASYNC_H

#include <string>
#include <memory>
#include <functional>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <unordered_map>

#include "nlp_engine.h"

namespace pce::nlp {

// ============ Stream callbacks ============

using StreamCallback = std::function<void(const std::string& chunk, bool is_final)>;
using CompletionCallback = std::function<void(const std::string& result, bool success)>;

// ============ Data Structures ============

// Async result wrapper
struct AsyncResult {
    std::string result;
    bool success;
    std::string error;
    std::string task_id;
};

// Async task manager
class AsyncTaskManager {
private:
    std::unordered_map<std::string, std::future<AsyncResult>> tasks_;
    std::vector<std::thread> workers_;
    std::mutex tasks_mutex_;

public:
    ~AsyncTaskManager() {
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }
    std::string submit_task(std::function<AsyncResult()> task);
    AsyncResult get_result(const std::string& task_id, bool wait = true);
    void cancel_task(const std::string& task_id);
    bool is_task_complete(const std::string& task_id);
};

// Core NLP Engine with async/stream support
class AsyncNLPPlugin {
protected:
    std::shared_ptr<NLPModel> model_;

public:
    explicit AsyncNLPPlugin(std::shared_ptr<NLPModel> model) : model_(model) {}
    virtual ~AsyncNLPPlugin() = default;

    // Async processing with streaming
    virtual std::string process_async_stream(
        const std::string& input,
        StreamCallback callback,
        const std::unordered_map<std::string, std::string>& options = {}
    ) = 0;

    // Traditional async processing
    virtual std::string process_async(
        const std::string& input,
        CompletionCallback callback,
        const std::unordered_map<std::string, std::string>& options = {}
    ) = 0;

    // Synchronous processing (for compatibility)
    virtual std::string process_sync(
        const std::string& input,
        const std::unordered_map<std::string, std::string>& options = {}
    ) = 0;
};

// Main async engine
class AsyncNLPEngine {
private:
    std::shared_ptr<NLPModel> model_;
    std::unique_ptr<AsyncTaskManager> task_manager_;
    std::unordered_map<std::string, std::shared_ptr<AsyncNLPPlugin>> plugins_;
    std::mutex plugins_mutex_;
    std::thread worker_thread_;
    bool is_running_;

public:
    explicit AsyncNLPEngine(std::shared_ptr<NLPModel> model);
    ~AsyncNLPEngine();

    bool initialize();
    bool shutdown();

    // Plugin management
    bool register_plugin(std::shared_ptr<AsyncNLPPlugin> plugin);
    bool unregister_plugin(const std::string& name);

    // Async processing
    std::string process_text_async(
        const std::string& text,
        const std::string& plugin_name,
        StreamCallback stream_callback = nullptr,
        const std::unordered_map<std::string, std::string>& options = {}
    );

    // Task management
    std::string submit_task(
        std::function<AsyncResult()> task,
        const std::string& task_name = ""
    );

    AsyncResult get_task_result(const std::string& task_id);

    // Streaming operations
    void stream_text(
        const std::string& text,
        const std::string& plugin_name,
        StreamCallback callback,
        const std::unordered_map<std::string, std::string>& options = {}
    );
};

} // namespace pce::nlp

#endif // NLP_ENGINE_ASYNC_H
