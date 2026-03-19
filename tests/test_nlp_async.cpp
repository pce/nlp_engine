#include "../nlp/nlp_engine_async.hh"
#include "CppUTest/TestHarness.h"
#include "CppUTest/CommandLineTestRunner.h"
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <filesystem>

using namespace pce::nlp;
namespace fs = std::filesystem;

/**
 * @file test_nlp_async.cpp
 * @brief Integration tests for AsyncNLPEngine using CppUTest.
 *
 * Verifies the asynchronous task manager, streaming callbacks,
 * and concurrency stability of the NLP engine.
 */

TEST_GROUP(AsyncNLPEngineTests) {
    std::shared_ptr<NLPModel> model;
    std::unique_ptr<AsyncNLPEngine> engine;

    void setup() {
        model = std::make_shared<NLPModel>();

        std::string data_path = "data";
        if (!fs::exists(data_path)) {
            data_path = "../data";
        }

        // We don't fail if model doesn't load, as some async tasks
        // might use addons or basic logic.
        model->load_from(data_path);

        engine = std::make_unique<AsyncNLPEngine>(model);
        engine->initialize();
    }

    void teardown() {
        engine->shutdown();
        engine.reset();
        model.reset();
    }
};

TEST(AsyncNLPEngineTests, SyncBridgeCall) {
    // Test that the async engine can still perform synchronous tasks
    std::string text = "This is a test of the synchronous bridge.";
    std::string result_json = engine->process_sync(text, "language");

    CHECK(!result_json.empty());
    // Should detect English
    CHECK(result_json.find("en") != std::string::npos);
}

TEST(AsyncNLPEngineTests, AsyncTaskSubmission) {
    std::string text = "Async processing should not block.";
    std::string task_id = engine->process_text_async(text, "default");

    CHECK(!task_id.empty());

    // Poll for result with a reasonable timeout
    AsyncResult res;
    bool completed = false;
    for (int i = 0; i < 20; ++i) {
        res = engine->get_task_result(task_id);
        if (res.success || !res.error.empty()) {
            completed = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    CHECK_TRUE(completed);
    CHECK_TRUE(res.success);
}

TEST(AsyncNLPEngineTests, StreamingInterface) {
    std::atomic<int> chunk_count{0};
    std::atomic<bool> is_finished{false};
    std::string accumulated_log = "";

    auto stream_callback = [&](const std::string& chunk, bool is_final) {
        chunk_count++;
        accumulated_log += chunk;
        if (is_final) {
            is_finished = true;
        }
    };

    std::string text = "Streaming analysis provides real-time feedback.";
    // Request POS tagging to ensure multiple chunks are generated
    engine->stream_text(text, "default", stream_callback, {{"pos_tagging", "true"}});

    // Wait for stream to complete with timeout (2 seconds)
    for (int i = 0; i < 40; ++i) {
        if (is_finished) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    CHECK_TRUE(is_finished);
    CHECK(chunk_count > 1);
    // The "default" async pipeline in nlp_engine_async.cpp ends with "Finished."
    CHECK(accumulated_log.find("Finished") != std::string::npos);
}

TEST(AsyncNLPEngineTests, ConcurrencyStability) {
    const int num_tasks = 4;
    std::vector<std::string> task_ids;

    // Submit multiple tasks rapidly
    for (int i = 0; i < num_tasks; ++i) {
        task_ids.push_back(engine->process_text_async("Concurrent text task " + std::to_string(i), "default"));
    }

    // Verify all tasks complete successfully
    for (const auto& id : task_ids) {
        // get_task_result defaults to blocking wait if not specified otherwise in implementation,
        // but our implementation returns immediately if not ready.
        // We use a simple loop to wait for each.
        AsyncResult res;
        bool success = false;
        for (int j = 0; j < 50; ++j) {
            res = engine->get_task_result(id);
            if (res.success) {
                success = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        CHECK_TRUE_TEXT(success, "One of the concurrent tasks failed to complete in time");
    }
}

int main(int ac, char** av) {
    // Memory leak detection is disabled for async tests to avoid conflicts
    // between CppUTest's new/delete macros and standard library threading (std::async).
    MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
    return CommandLineTestRunner::RunAllTests(ac, av);
}
