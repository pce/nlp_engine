#include "nlp_engine_async.hh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>

namespace py = pybind11;
using namespace pce::nlp;

/**
 * @class PythonAsyncNLPEngine
 * @brief Python-friendly wrapper for the C++ AsyncNLPEngine.
 *
 * This class manages the lifecycle of the NLPModel and AsyncNLPEngine,
 * providing a simplified interface for Python bindings.
 */
class PythonAsyncNLPEngine {
private:
    std::shared_ptr<NLPModel> model_;
    std::unique_ptr<AsyncNLPEngine> engine_;

public:
    PythonAsyncNLPEngine() {
        // Initialize model and engine
        model_ = std::make_shared<NLPModel>();
        engine_ = std::make_unique<AsyncNLPEngine>(model_);
        engine_->initialize();
    }

    ~PythonAsyncNLPEngine() {
        shutdown();
    }

    void initialize() {
        if (engine_) {
            engine_->initialize();
        }
    }

    void shutdown() {
        if (engine_) {
            engine_->shutdown();
        }
    }

    /**
     * @brief Load resources for the underlying model.
     * @param path Directory containing linguistic resources.
     * @return true if successful.
     */
    bool load_model(const std::string& path) {
        if (model_) {
            return model_->load_from(path);
        }
        return false;
    }

    /**
     * @brief Synchronous processing (blocking).
     */
    /**
     * @brief Synchronous granular processing.
     * @param text The input text.
     * @param method The specific NLP method to invoke (e.g., 'spell_check', 'sentiment').
     * @param options Method-specific options.
     * @return JSON string containing the results.
     */
    std::string process_sync(
        const std::string& text,
        const std::string& method,
        const std::unordered_map<std::string, std::string>& options = {}
    ) {
        if (engine_) {
            return engine_->process_sync(text, method, options);
        }
        return "{\"error\": \"Engine not initialized\"}";
    }

    std::string process_text_sync(
        const std::string& text,
        const std::string& plugin_name,
        const std::unordered_map<std::string, std::string>& options = {}
    ) {
        return process_sync(text, plugin_name, options);
    }

    /**
     * @brief Submit an asynchronous task.
     * @return Task ID string.
     */
    std::string process_text_async(
        const std::string& text,
        const std::string& plugin_name,
        const std::unordered_map<std::string, std::string>& options = {}
    ) {
        return engine_->process_text_async(text, plugin_name, nullptr, options);
    }

    /**
     * @brief Stream results back to a Python callback.
     */
    void stream_text(
        const std::string& text,
        const std::string& plugin_name,
        py::function callback,
        const std::unordered_map<std::string, std::string>& options = {}
    ) {
        auto stream_callback = [callback](const std::string& chunk, bool is_final) {
            // Ensure we have the GIL before calling back into Python
            py::gil_scoped_acquire acquire;
            try {
                // Convert to Python string. py::str(std::string) handles UTF-8.
                callback(py::str(chunk), is_final);
            } catch (const py::error_already_set&) {
                // If decoding fails, fallback to bytes or a "replace" strategy to prevent crashes
                try {
                    py::bytes b(chunk);
                    callback(b.attr("decode")("utf-8", "replace"), is_final);
                } catch (...) {
                    // Ultimate fallback for corrupted data
                    callback(" [Data Error] ", is_final);
                }
            }
        };

        // Release the GIL to allow the C++ thread to run without blocking the Python event loop
        py::gil_scoped_release release;
        engine_->stream_text(text, plugin_name, stream_callback, options);
    }

    bool is_ready() {
        return engine_ != nullptr && model_ != nullptr && model_->is_ready();
    }
};

PYBIND11_MODULE(nlp_engine, m) {
    m.doc() = "NLP Engine with async/stream support (pce::nlp)";

    py::class_<PythonAsyncNLPEngine>(m, "AsyncNLPEngine")
        .def(py::init<>())
        .def("load_model", &PythonAsyncNLPEngine::load_model, "Load model resources from path")
        .def("initialize", &PythonAsyncNLPEngine::initialize, "Initialize the engine")
        .def("shutdown", &PythonAsyncNLPEngine::shutdown, "Shutdown the engine")
        .def("process_sync", &PythonAsyncNLPEngine::process_sync, py::arg("text"), py::arg("method"), py::arg("options") = std::unordered_map<std::string, std::string>())
        .def("process_text_sync", &PythonAsyncNLPEngine::process_text_sync, py::arg("text"), py::arg("plugin_name"), py::arg("options") = std::unordered_map<std::string, std::string>())
        .def("process_text_async", &PythonAsyncNLPEngine::process_text_async)
        .def("stream_text", &PythonAsyncNLPEngine::stream_text)
        .def("is_ready", &PythonAsyncNLPEngine::is_ready);
}
