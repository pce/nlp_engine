#include "nlp_engine_async.hh"
#include "addons/markov_addon.hh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace pce::nlp;

/**
 * @class PythonAsyncNLPEngine
 * @brief Python-friendly wrapper for the C++ AsyncNLPEngine.
 */
class PythonAsyncNLPEngine {
private:
    std::shared_ptr<NLPModel> model_;
    std::unique_ptr<AsyncNLPEngine> engine_;

public:
    PythonAsyncNLPEngine() {
        model_ = std::make_shared<NLPModel>();
        engine_ = std::make_unique<AsyncNLPEngine>(model_);
        // Explicitly initialize so status checks pass immediately in Python
        engine_->initialize();
    }

    ~PythonAsyncNLPEngine() {
        shutdown();
    }

    void initialize() {
        if (engine_) engine_->initialize();
    }

    void shutdown() {
        if (engine_) engine_->shutdown();
    }

    bool load_model(const std::string& path) {
        return model_ && model_->load_from(path);
    }

    /**
     * @brief Directly register a MarkovAddon instance from Python.
     */
    bool register_markov_addon(std::shared_ptr<MarkovAddon> addon, const std::string& name = "") {
        if (!engine_ || !addon) return false;
        if (!name.empty()) {
            // We need a way to override the name for the engine's map
            // Since name() is virtual, we can't easily change it without a wrapper
            // But AsyncNLPEngine uses the addon->name() by default.
            // For now, we'll implement a name override in AsyncNLPEngine if needed,
            // or assume the addon instance is already configured.
        }
        return engine_->add_addon(addon);
    }

    /**
     * @brief Convenience method to load and register a Markov model in one shot.
     */
    bool load_markov_model(const std::string& model_path, const std::string& name = "") {
        if (!engine_) return false;
        auto markov = std::make_shared<MarkovAddon>();
        if (markov->load_knowledge_pack(model_path)) {
            if (!name.empty()) {
                markov->set_name(name);
            }
            return engine_->add_addon(markov);
        }
        return false;
    }

    std::string process_sync(
        const std::string& text,
        const std::string& method,
        const std::unordered_map<std::string, std::string>& options = {},
        const std::string& session_id = ""
    ) {
        if (engine_) return engine_->process_sync(text, method, options, session_id);
        return "{\"error\": \"Engine not initialized\"}";
    }

    std::string process_text_async(
        const std::string& text,
        const std::string& addon_name,
        const std::unordered_map<std::string, std::string>& options = {},
        const std::string& session_id = ""
    ) {
        if (!engine_) return "";
        return engine_->process_text_async(text, addon_name, nullptr, options, session_id);
    }

    void stream_text(
        const std::string& text,
        const std::string& addon_name,
        py::function callback,
        const std::unordered_map<std::string, std::string>& options = {},
        const std::string& session_id = ""
    ) {
        auto stream_callback = [callback](const std::string& chunk, bool is_final) {
            py::gil_scoped_acquire acquire;
            try {
                callback(py::str(chunk), is_final);
            } catch (const py::error_already_set&) {
                try {
                    py::bytes b(chunk);
                    callback(b.attr("decode")("utf-8", "replace"), is_final);
                } catch (...) {
                    callback(" [Data Error] ", is_final);
                }
            }
        };

        py::gil_scoped_release release;
        engine_->stream_text(text, addon_name, stream_callback, options, session_id);
    }

    void clear_session(const std::string& session_id) {
        if (engine_) engine_->clear_context(session_id);
    }

    bool has_addon(const std::string& name) {
        return engine_ && engine_->has_addon(name);
    }

    bool remove_addon(const std::string& name) {
        return engine_ && engine_->remove_addon(name);
    }

    bool is_ready() {
        return engine_ != nullptr && model_ != nullptr && model_->is_ready();
    }
};

PYBIND11_MODULE(nlp_engine, m) {
    m.doc() = "NLP Engine with Addon support for Python (pce::nlp)";

    // --- Context Bindings ---
    py::class_<AddonContext, std::shared_ptr<AddonContext>>(m, "AddonContext")
        .def_readwrite("session_id", &AddonContext::session_id)
        .def_readwrite("metadata", &AddonContext::metadata)
        .def_readwrite("history", &AddonContext::history);

    // --- Markov Addon Bindings ---
    py::class_<MarkovAddon, std::shared_ptr<MarkovAddon>>(m, "MarkovAddon")
        .def(py::init<>())
        .def_property_readonly("name", &MarkovAddon::name)
        .def_property_readonly("version", &MarkovAddon::version)
        .def("is_ready", &MarkovAddon::is_ready)
        .def("load_knowledge_pack", &MarkovAddon::load_knowledge_pack, py::arg("path"),
             "Load a pre-trained JSON Knowledge Pack")
        .def("train", &MarkovAddon::train, py::arg("source_path"), py::arg("output_path"),
             "Train a new model from a text file")
        .def("get_training_progress", &MarkovAddon::get_training_progress)
        .def("process", [](MarkovAddon& self, const std::string& input,
                           const std::unordered_map<std::string, std::string>& options,
                           std::shared_ptr<AddonContext> context) {
            auto resp = self.process(input, options, context);
            py::dict d;
            d["output"] = resp.output;
            d["success"] = resp.success;
            d["error"] = resp.error_message;
            return d;
        }, py::arg("input"),
           py::arg("options") = std::unordered_map<std::string, std::string>(),
           py::arg("context") = nullptr);

    // --- Main Engine Bindings ---
    py::class_<PythonAsyncNLPEngine>(m, "AsyncNLPEngine")
        .def(py::init<>())
        .def("load_model", &PythonAsyncNLPEngine::load_model, "Load base linguistic resources")
        .def("initialize", &PythonAsyncNLPEngine::initialize)
        .def("shutdown", &PythonAsyncNLPEngine::shutdown)
        // Addon registration
        .def("register_markov_addon", &PythonAsyncNLPEngine::register_markov_addon,
             py::arg("addon"), py::arg("name") = "",
             "Register a pre-configured MarkovAddon instance")
        .def("load_markov_model", &PythonAsyncNLPEngine::load_markov_model,
             py::arg("model_path"), py::arg("name") = "",
             "Quick-load and register a Markov model from path")
        // Processing
        .def("process_sync", &PythonAsyncNLPEngine::process_sync,
             py::arg("text"), py::arg("method"),
             py::arg("options") = std::unordered_map<std::string, std::string>(),
             py::arg("session_id") = "")
        .def("process_text_async", &PythonAsyncNLPEngine::process_text_async,
             py::arg("text"), py::arg("addon_name"),
             py::arg("options") = std::unordered_map<std::string, std::string>(),
             py::arg("session_id") = "")
        .def("stream_text", &PythonAsyncNLPEngine::stream_text,
             py::arg("text"), py::arg("addon_name"), py::arg("callback"),
             py::arg("options") = std::unordered_map<std::string, std::string>(),
             py::arg("session_id") = "")
        // Utility
        .def("clear_session", &PythonAsyncNLPEngine::clear_session, py::arg("session_id"))
        .def("has_addon", &PythonAsyncNLPEngine::has_addon)
        .def("remove_addon", &PythonAsyncNLPEngine::remove_addon)
        .def("is_ready", &PythonAsyncNLPEngine::is_ready);
}
