#include "rll_policy/inference_engine.hpp"

#include <torch/cuda.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace rll_policy {
struct InferenceEngine::Model {
  PolicyModelConfig config;
  torch::jit::Module module;
};

namespace {
torch::Device parse_device(const std::string& requested) {
  if (requested == "cpu") return torch::Device(torch::kCPU);
  if (requested == "cuda") {
    if (!torch::cuda::is_available()) throw std::runtime_error("CUDA was configured but is not available");
    return torch::Device(torch::kCUDA, 0);
  }
  if (requested.rfind("cuda:", 0) == 0) {
    std::size_t consumed = 0;
    int index = -1;
    try { index = std::stoi(requested.substr(5), &consumed); }
    catch (...) { throw std::invalid_argument("invalid device '" + requested + "'"); }
    if (consumed != requested.size() - 5 || index < 0) throw std::invalid_argument("invalid device '" + requested + "'");
    if (!torch::cuda::is_available() || index >= torch::cuda::device_count())
      throw std::runtime_error("configured device '" + requested + "' is unavailable");
    return torch::Device(torch::kCUDA, index);
  }
  throw std::invalid_argument("device must be 'cpu', 'cuda', or 'cuda:<index>'");
}

void validate_tensor(const torch::Tensor& value, const torch::Device& device,
                     std::int64_t width, const std::string& description) {
  if (!value.defined()) throw std::runtime_error(description + " is undefined");
  if (!value.is_floating_point() || value.scalar_type() != torch::kFloat32)
    throw std::runtime_error(description + " must have float32 dtype");
  if (value.dim() != 2 || value.size(0) != 1 || value.size(1) != width)
    throw std::runtime_error(description + " must have shape [1, " + std::to_string(width) + "]");
  if (value.device() != device) throw std::runtime_error(description + " is on the wrong device");
  if (!torch::isfinite(value).all().item<bool>()) throw std::runtime_error(description + " contains NaN or Inf");
}

template <typename ModelType>
torch::Tensor forward(ModelType& model, const torch::Tensor& observation, const torch::Device& device) {
  auto input = observation.to(device, torch::kFloat32).contiguous();
  torch::NoGradGuard no_grad;
  auto value = model.module.forward({input});
  if (!value.isTensor()) throw std::runtime_error("TorchScript output must be a tensor");
  auto output = value.toTensor();
  validate_tensor(output, device, model.config.action_dim, "TorchScript output");
  return output;
}
}

InferenceEngine::InferenceEngine() = default;
InferenceEngine::~InferenceEngine() = default;
InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;
InferenceEngine& InferenceEngine::operator=(InferenceEngine&&) noexcept = default;

void InferenceEngine::load_models(const std::vector<PolicyModelConfig>& configs, const std::string& device) {
  if (configs.empty()) throw std::invalid_argument("at least one policy model must be configured");
  auto selected_device = parse_device(device);
  std::unordered_map<std::string, std::unique_ptr<Model>> loaded;
  for (const auto& config : configs) {
    if (config.name.empty()) throw std::invalid_argument("policy model name cannot be empty");
    if (config.observation_dim <= 0 || config.action_dim <= 0)
      throw std::invalid_argument("policy '" + config.name + "' dimensions must be positive");
    if (loaded.count(config.name)) throw std::invalid_argument("duplicate policy model name '" + config.name + "'");
    if (!std::filesystem::is_regular_file(config.checkpoint))
      throw std::runtime_error("checkpoint for policy '" + config.name + "' does not exist: " + config.checkpoint.string());
    try {
      auto entry = std::make_unique<Model>();
      entry->config = config;
      entry->module = torch::jit::load(config.checkpoint.string(), selected_device);
      entry->module.eval();
      auto probe = torch::zeros({1, config.observation_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(selected_device));
      auto output = forward(*entry, probe, selected_device);
      (void)output;
      loaded.emplace(config.name, std::move(entry));
    } catch (const std::exception& error) {
      throw std::runtime_error("failed to initialize policy '" + config.name + "' from '" + config.checkpoint.string() + "': " + error.what());
    }
  }
  models_.swap(loaded);
  device_ = selected_device;
}

torch::Tensor InferenceEngine::infer(const std::string& policy_name, const torch::Tensor& observation) {
  auto found = models_.find(policy_name);
  if (found == models_.end()) throw std::runtime_error("unknown policy model '" + policy_name + "'");
  if (!observation.defined() || !observation.is_floating_point() || observation.scalar_type() != torch::kFloat32)
    throw std::runtime_error("observation for policy '" + policy_name + "' must be a defined float32 tensor");
  const auto obs_dim = found->second->config.observation_dim;
  if (observation.dim() != 2 || observation.size(0) != 1 || observation.size(1) != obs_dim)
    throw std::runtime_error("observation for policy '" + policy_name + "' must have shape [1, " + std::to_string(obs_dim) + "]");
  if (!torch::isfinite(observation).all().item<bool>())
    throw std::runtime_error("observation for policy '" + policy_name + "' contains NaN or Inf");
  try {
    return forward(*found->second, observation, device_);
  } catch (const std::exception& error) {
    throw std::runtime_error("inference failed for policy '" + policy_name + "' from '" +
                             found->second->config.checkpoint.string() + "': " + error.what());
  }
}
}
