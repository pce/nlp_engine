#include <catch2/catch_all.hpp>
#include "nlp/nlp_engine.hh"
#include "nlp/addons/onnx_addon.hh"
#include <memory>
#include <string>

using namespace pce::nlp;


TEST_CASE("ONNXAddon: Load Transformer", "[onnx][transformer]") {
    auto model = std::make_shared<NLPModel>();
    NLPEngine engine(model);
    ONNXAddon onnx_addon;
    engine.addAddon(onnx_addon);
    engine.loadModel();
}
