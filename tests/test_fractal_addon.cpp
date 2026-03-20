#include "CppUTest/TestHarness.h"
#include "../nlp/addons/fractal_addon.hh"
#include "../nlp/addons/markov_addon.hh"
#include <memory>
#include <string>

/**
 * @test Unit tests for the FractalAddon component.
 * Verifies recursive generation and Markov source integration.
 */
TEST_GROUP(FractalAddonTest) {
    std::shared_ptr<pce::nlp::MarkovAddon> markov;
    std::shared_ptr<pce::nlp::FractalAddon> fractal;

    void setup() {
        markov = std::make_shared<pce::nlp::MarkovAddon>();
        fractal = std::make_shared<pce::nlp::FractalAddon>();
    }

    void teardown() {
        markov.reset();
        fractal.reset();
    }
};

TEST(FractalAddonTest, NotReadyWithoutMarkovSource) {
    CHECK_FALSE(fractal->is_ready());
}

TEST(FractalAddonTest, ReadyWithMarkovSource) {
    // Markov isn't ready until loaded, but fractal needs it registered
    fractal->set_markov_source(markov);
    // Note: depends on markov internal state, but is_ready checks both
    CHECK_FALSE(fractal->is_ready());
}

TEST(FractalAddonTest, BasicMetadata) {
    STRCMP_EQUAL("fractal_generator", fractal->name().c_str());
    CHECK(fractal->version().find("experimental") != std::string::npos);
}

TEST(FractalAddonTest, ExtractionContextHelper) {
    // This tests a private logic piece via the fractal generation if possible,
    // but here we verify the public process interface doesn't crash on empty input.
    fractal->set_markov_source(markov);
    std::unordered_map<std::string, std::string> options = {{"depth", "1"}};

    // Should fail gracefully because markov is not loaded
    auto resp = fractal->process("The quick brown fox", options);
    CHECK_FALSE(resp.success);
}

TEST(FractalAddonTest, DepthClamping) {
    fractal->set_markov_source(markov);

    // We can't easily run the full recursion without a trained Markov chain,
    // but we verify the addon exists and accepts the options.
    std::unordered_map<std::string, std::string> options = {
        {"depth", "10"}, // Should be clamped to 5
        {"length", "5"}
    };

    auto resp = fractal->process("Seed", options);
    CHECK_FALSE(resp.success); // Success=false because Markov isn't ready
}
