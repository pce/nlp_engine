#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <filesystem>
#include "../nlp/nlp_engine_async.hh"

using namespace pce::nlp;
namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;

/**
 * @file test_nlp_async.cpp
 * @brief Integration tests for AsyncNLPEngine using Catch2 v3.
 *
 * Verifies the asynchronous task manager, streaming callbacks,
 * and concurrency stability of the NLP engine using modern C++23 patterns.
 */

struct AsyncNLPEngineFixture {
    std::shared_ptr<NLPModel> model;
    std::unique_ptr<AsyncNLPEngine> engine;

    AsyncNLPEngineFixture() {
        model = std::make_shared<NLPModel>();
        std::string data_path = "data";
        if (!fs::exists(data_path)) {
            data_path = "../data";
        }
        model->load_from(data_path);
        engine = std::make_unique<AsyncNLPEngine>(model);
        engine->initialize();
    }

    ~AsyncNLPEngineFixture() {
        engine->shutdown();
    }
};

TEST_CASE("AsyncNLPEngine Operations", "[async][nlp]") {
    AsyncNLPEngineFixture fix;

    SECTION("Sync Bridge Call") {
        // Test that the async engine can still perform synchronous tasks
        std::string text = "This is a test of the synchronous bridge.";
        std::string result_json = fix.engine->process_sync(text, "language");

        CHECK_FALSE(result_json.empty());
        // Should detect English
        CHECK_THAT(result_json, ContainsSubstring("en"));
    }

    SECTION("Async Task Submission") {
        std::string text = "Async processing should not block.";
        std::string task_id = fix.engine->process_text_async(text, "default");

        REQUIRE_FALSE(task_id.empty());

        // Poll for result with a reasonable timeout
        AsyncResult res;
        bool completed = false;
        for (int i = 0; i < 40; ++i) {
            res = fix.engine->get_task_result(task_id);
            if (res.success || !res.error.empty()) {
                completed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        CHECK(completed);
        CHECK(res.success);
    }

    SECTION("Streaming Interface") {
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
        fix.engine->stream_text(text, "default", stream_callback, {{"pos_tagging", "true"}});

        // Wait for stream to complete with timeout (2 seconds)
        for (int i = 0; i < 40; ++i) {
            if (is_finished) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        CHECK(is_finished);
        CHECK(chunk_count > 0);
        // Ensure the log contains fundamental completion markers
        CHECK_THAT(accumulated_log, ContainsSubstring("Finished") || ContainsSubstring("Success"));
    }

    SECTION("Concurrency Stability") {
        const int num_tasks = 4;
        std::vector<std::string> task_ids;

        // Submit multiple tasks rapidly
        for (int i = 0; i < num_tasks; ++i) {
            task_ids.push_back(fix.engine->process_text_async("Concurrent task " + std::to_string(i), "default"));
        }

        // Verify all tasks complete successfully
        for (const auto& id : task_ids) {
            AsyncResult res;
            bool success = false;
            for (int j = 0; j < 50; ++j) {
                res = fix.engine->get_task_result(id);
                if (res.success) {
                    success = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            INFO("Checking task ID: " << id);
            CHECK(success);
        }
    }
}
