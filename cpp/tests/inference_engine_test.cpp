#include "rll_policy/inference_engine.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
template <typename F> bool rejects(F&& f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}
rll_policy::PolicyModelConfig cfg(const std::filesystem::path& dir, std::string name,
                                  std::string file, std::int64_t obs, std::int64_t action) {
  return {std::move(name), dir / file, obs, action};
}
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path dir(argv[1]);
  rll_policy::InferenceEngine engine;
  engine.load_models({cfg(dir,"velocity","velocity.pt",3,2),
                      cfg(dir,"tracking","tracking.pt",5,4)}, "cpu");
  auto velocity = engine.infer("velocity", torch::tensor({{4.0F, 7.0F, 9.0F}}));
  assert(velocity.sizes() == torch::IntArrayRef({1, 2}));
  assert(velocity[0][0].item<float>() == 5.0F && velocity[0][1].item<float>() == 8.0F);
  auto tracking = engine.infer("tracking", torch::zeros({1, 5}));
  assert(tracking.sizes() == torch::IntArrayRef({1, 4}));
  assert(torch::allclose(tracking, torch::ones({1, 4})));
  assert(rejects([&] { (void)engine.infer("missing", torch::zeros({1, 3})); }));
  assert(rejects([&] { (void)engine.infer("velocity", torch::zeros({1, 4})); }));
  assert(rejects([&] { (void)engine.infer("velocity", torch::full({1, 3}, NAN)); }));

  auto invalid_model = [&](std::string name, std::string file, std::int64_t expected_actions) {
    rll_policy::InferenceEngine invalid;
    return rejects([&] { invalid.load_models({cfg(dir,std::move(name),std::move(file),3,expected_actions)}, "cpu"); });
  };
  assert(invalid_model("tuple", "tuple.pt", 2));
  assert(invalid_model("shape", "wrong_shape.pt", 3));
  assert(invalid_model("dtype", "wrong_dtype.pt", 2));
  assert(invalid_model("nan", "non_finite.pt", 2));
  assert(invalid_model("corrupt", "corrupt.pt", 2));
  assert(invalid_model("missing", "missing.pt", 2));
  assert(rejects([&] { rll_policy::InferenceEngine unavailable; unavailable.load_models({cfg(dir,"cuda","velocity.pt",3,2)}, "cuda:99"); }));
  assert(rejects([&] { rll_policy::InferenceEngine duplicate; duplicate.load_models({cfg(dir,"x","velocity.pt",3,2),cfg(dir,"x","tracking.pt",5,4)}, "cpu"); }));
}
