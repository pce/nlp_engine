#ifndef NLP_ADDON_SYSTEM_HH
#define NLP_ADDON_SYSTEM_HH

#include <string>
#include <memory>
#include <vector>
#include <variant>
#include <unordered_map>
#include <optional>
#include <functional>

/**
 * @file nlp_addon_system.hh
 * @brief Zero-overhead static addon architecture for the NLP engine.
 *
 * This system uses CRTP (Curiously Recurring Template Pattern) and
 * polymorphism to provide a plugin-like architecture without the
 * runtime overhead of heavy virtual registries.
 */

namespace pce::nlp {

/**
 * @struct AddonContext
 * @brief Persistent state for a specific session or document.
 */
struct AddonContext {
    std::string session_id;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<std::string> history;
};

/**
 * @struct AddonResponse
 * @brief Standardized result from any NLP Addon operation.
 */
struct AddonResponse {
    std::string output;                             ///< The primary output (text or JSON).
    bool success = false;                           ///< Status of the operation.
    std::string error_message;                      ///< Diagnostic info if success is false.
    std::unordered_map<std::string, double> metrics; ///< Performance or logic metrics.
    std::unordered_map<std::string, std::string> metadata; ///< Key-value pairs for structured response data.
};

/**
 * @class INLPAddon
 * @brief Abstract base class to allow pointer-based registration and polymorphism.
 */
class INLPAddon {
public:
    virtual ~INLPAddon() = default;
    virtual const std::string& name() const = 0;
    virtual const std::string& version() const = 0;
    virtual bool initialize() = 0;
    virtual bool is_ready() const = 0;
    virtual AddonResponse process(const std::string& input,
                                 const std::unordered_map<std::string, std::string>& options = {},
                                 std::shared_ptr<AddonContext> context = nullptr) = 0;

    virtual void process_stream(const std::string& input,
                               std::function<void(const std::string& chunk, bool is_final)> callback,
                               const std::unordered_map<std::string, std::string>& options = {},
                               std::shared_ptr<AddonContext> context = nullptr) = 0;
};

/**
 * @class NLPAddon
 * @brief CRTP-based Base class for zero-overhead static Addons.
 *
 * Implements the INLPAddon interface to support shared pointers in maps while
 * preserving the static dispatch benefits of CRTP where possible.
 */
template <typename Derived>
class NLPAddon : public INLPAddon {
public:
    const std::string& name() const override {
        return static_cast<const Derived*>(this)->name_impl();
    }

    const std::string& version() const override {
        return static_cast<const Derived*>(this)->version_impl();
    }

    bool initialize() override {
        return static_cast<Derived*>(this)->init_impl();
    }

    AddonResponse process(const std::string& input,
                         const std::unordered_map<std::string, std::string>& options = {},
                         std::shared_ptr<AddonContext> context = nullptr) override {
        return static_cast<Derived*>(this)->process_impl(input, options, context);
    }

    void process_stream(const std::string& input,
                        std::function<void(const std::string& chunk, bool is_final)> callback,
                        const std::unordered_map<std::string, std::string>& options = {},
                        std::shared_ptr<AddonContext> context = nullptr) override {
        static_cast<Derived*>(this)->process_stream_impl(input, callback, options, context);
    }

protected:
    ~NLPAddon() = default;
};

// --- Addon Collection ---

// Forward declarations
class MarkovAddon;

/**
 * @typedef AddonVariant
 * @brief A type-safe container for any supported NLP Addon.
 *
 * While we use INLPAddon* for registration, this variant remains useful
 * for stack-based visitors or strict type checking.
 */
using AddonVariant = std::variant<
    std::shared_ptr<MarkovAddon>
>;

/**
 * @struct AddonVisitor
 * @brief Specialized visitor to invoke addon logic.
 */
struct AddonVisitor {
    const std::string& input;
    const std::unordered_map<std::string, std::string>& options;

    template <typename T>
    AddonResponse operator()(const std::shared_ptr<T>& addon) const {
        if (!addon || !addon->is_ready()) {
            return {"", false, "Addon not ready or null", {}};
        }
        return addon->process(input, options);
    }

    // Support for the base pointer interface
    AddonResponse operator()(const std::shared_ptr<INLPAddon>& addon) const {
        if (!addon || !addon->is_ready()) {
            return {"", false, "Addon not ready or null", {}};
        }
        return addon->process(input, options);
    }
};

/**
 * @interface ITrainable
 * @brief Optional interface for Addons that support the separate Training Pipeline.
 */
class ITrainable {
public:
    virtual ~ITrainable() = default;

    virtual bool train(const std::string& source_path, const std::string& model_output_path) = 0;
    virtual float get_training_progress() const = 0;
};

} // namespace pce::nlp

#endif // NLP_ADDON_SYSTEM_HH
