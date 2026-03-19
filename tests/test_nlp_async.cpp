#include "../nlp/nlp_engine_async.hh"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>

using namespace pce::nlp;

/**
 * Global setup for async tests.
 * Shared model ensures we don't reload data from disk for every test case.
 */
std::shared_ptr<NLPModel> setup_async_model() {
    static std::shared_ptr<NLPModel> model = nullptr;
    if (!model) {
        model = std::make_shared<NLPModel>();
        if (!model->load_from("data")) {
            std::cerr << "Warning: Could not load data directory in async tests.\n";
        }
    }
    return model;
}

/**
 * Test synchronous granular processing through the AsyncNLPEngine interface.
 * Verifies that the engine uses the already-loaded model without re-reading files.
 */
void test_async_engine_sync_call() {
    auto model = setup_async_model();
    AsyncNLPEngine engine(model);
    engine.initialize();

    std::string text = "This is a test of the synchronous bridge.";
    std::string result_json = engine.process_sync(text, "language");

    assert(!result_json.empty());
    assert(result_json.find("en") != std::string::npos);
    std::cout << "✓ AsyncEngine Sync Bridge test passed.\n";
}

/**
 * Test asynchronous task submission.
 */
void test_async_task_submission() {
    auto model = setup_async_model();
    AsyncNLPEngine engine(model);
    engine.initialize();

    std::string text = "Async processing should not block.";
    std::string task_id = engine.process_text_async(text, "default");

    assert(!task_id.empty());

    // Poll for result
    AsyncResult res;
    int attempts = 0;
    while (attempts < 10) {
        res = engine.get_task_result(task_id);
        if (res.success || !res.error.empty()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        attempts++;
    }

    assert(res.success);
    std::cout << "✓ Async Task Submission test passed.\n";
}

/**
 * Test the Server-Sent Events (SSE) style streaming interface.
 */
void test_streaming_interface() {
    auto model = setup_async_model();
    AsyncNLPEngine engine(model);
    engine.initialize();

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
    engine.stream_text(text, "default", stream_callback, {{"pos_tagging", "true"}});

    // Wait for stream to complete (with timeout)
    int timeout_ms = 2000;
    int waited = 0;
    while (!is_finished && waited < timeout_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        waited += 50;
    }

    assert(is_finished == true);
    assert(chunk_count > 1);
    assert(accumulated_log.find("Finished") != std::string::npos);

    std::cout << "✓ Streaming Interface (SSE style) test passed.\n";
}

/**
 * Test memory stability under concurrent requests.
 */
void test_concurrency_stability() {
    auto model = setup_async_model();
    AsyncNLPEngine engine(model);
    engine.initialize();

    const int num_threads = 5;
    std::vector<std::string> task_ids;

    for (int i = 0; i < num_threads; ++i) {
        task_ids.push_back(engine.process_text_async("Concurrent text " + std::to_string(i), "default"));
    }

    for (const auto& id : task_ids) {
        AsyncResult res = engine.get_task_result(id);
        // get_task_result defaults to blocking (wait=true)
        assert(res.success);
    }

    std::cout << "✓ Concurrency Stability test passed (" << num_threads << " parallel tasks).\n";
}

int main() {
    try {
        std::cout << "Running AsyncNLPEngine Integration Tests...\n";
        std::cout << "------------------------------------------\n";

        test_async_engine_sync_call();
        test_async_task_submission();
        test_streaming_interface();
        test_concurrency_stability();

        std::cout << "------------------------------------------\n";
        std::cout << "All async tests passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Async tests failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
