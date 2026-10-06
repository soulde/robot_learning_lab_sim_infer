#pragma once

#include "rll_policy/inference_engine.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace rll_policy {

enum class Reliability { best_effort, reliable };

struct DdsRuntimeConfig {
  std::uint32_t domain_id{};
  std::string cyclonedds_uri;
  std::string state_topic{"robot/state"};
  std::string rc_topic{"robot/rc_command"};
  std::string motor_command_topic{"robot/motor_command"};
  Reliability sensor_reliability{Reliability::best_effort};
  Reliability command_reliability{Reliability::reliable};
};

struct PolicySlot {
  std::uint16_t index{};
  PolicyModelConfig model;
};

struct StateMachineConfig {
  std::string initial_state;
  std::map<std::string, std::optional<std::string>> state_policies;
  std::map<std::string, std::map<std::string, std::string>> transitions;
  std::map<std::uint16_t, std::string> mode_events;
  std::map<std::uint32_t, std::string> button_events;
};

struct PolicyRuntimeConfig {
  DdsRuntimeConfig dds;
  double policy_hz{50.0};
  double state_timeout_s{0.25};
  double rc_timeout_s{0.5};
  double transition_duration_s{0.5};
  double damping_kd{0.5};
  std::string device{"cpu"};
  std::filesystem::path plugin_path;
  std::filesystem::path plugin_config;
  std::vector<PolicySlot> policies;
  StateMachineConfig state_machine;

  static PolicyRuntimeConfig load(const std::filesystem::path& path);
};

class RuntimeStateMachine {
 public:
  explicit RuntimeStateMachine(StateMachineConfig config);
  const std::string& state() const noexcept { return state_; }
  std::optional<std::string> policy() const;
  bool apply(const std::string& event);
  bool on_mode(std::uint16_t mode);
  bool on_button(std::uint32_t previous_mask, std::uint32_t current_mask);
 private:
  StateMachineConfig config_;
  std::string state_;
};

}  // namespace rll_policy
