#pragma once

#include <torch/script.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rll_policy {

struct PolicyModelConfig {
  std::string name;
  std::filesystem::path checkpoint;
  std::int64_t observation_dim{};
  std::int64_t action_dim{};
};

class InferenceEngine {
 public:
  InferenceEngine();
  ~InferenceEngine();
  InferenceEngine(InferenceEngine&&) noexcept;
  InferenceEngine& operator=(InferenceEngine&&) noexcept;
  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;

  void load_models(const std::vector<PolicyModelConfig>& configs, const std::string& device);
  torch::Tensor infer(const std::string& policy_name, const torch::Tensor& observation);

 private:
  struct Model;
  torch::Device device_{torch::kCPU};
  std::unordered_map<std::string, std::unique_ptr<Model>> models_;
};

}  // namespace rll_policy
