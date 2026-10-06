#pragma once

#include "rll_policy/messages.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <string_view>

namespace rll_policy {

class InputProcessor {
 public:
  virtual ~InputProcessor() = default;
  virtual void on_policy_selected(std::string_view policy_type) { (void)policy_type; }
  virtual torch::Tensor build_observation(std::string_view policy_type, const RobotState& state,
                                         const RCCommand& rc) = 0;
};

using MotorCommandWriteFn = void (*)(void* context, const MotorCommand* command);

class OutputProcessor {
 public:
  virtual ~OutputProcessor() = default;
  virtual MotorCommand make_command(std::string_view policy_type, const torch::Tensor& action,
                                    const RobotState& state, std::uint64_t timestamp_ns) = 0;
  virtual MotorCommand make_fixed_pose_command(const RobotState& state,
                                                std::uint64_t timestamp_ns) = 0;
  virtual void publish(const MotorCommand& command) = 0;
};

}  // namespace rll_policy
