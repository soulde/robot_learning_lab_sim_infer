#include "rll_policy/plugin_api.hpp"

#include <torch/torch.h>

#include <stdexcept>
#include <string>

namespace {
class TestInput final : public rll_policy::InputProcessor {
 public:
  torch::Tensor build_observation(std::string_view, const rll_policy::RobotState& state,
                                  const rll_policy::RCCommand& rc) override {
    if (state.joint_position.empty()) throw std::invalid_argument("expected at least one joint");
    return torch::tensor({{static_cast<float>(state.joint_position[0]), rc.vx}}, torch::kFloat32);
  }
};
class TestOutput final : public rll_policy::OutputProcessor {
 public:
  TestOutput(rll_policy::MotorCommandWriteFn writer, void* context) : writer_(writer), context_(context) {}
  rll_policy::MotorCommand make_command(std::string_view, const torch::Tensor& action,
                                       const rll_policy::RobotState& state, std::uint64_t stamp) override {
    auto values=action.to(torch::kCPU).to(torch::kFloat64).contiguous().view(-1);
    if (values.numel() != 2) throw std::invalid_argument("expected two actions");
    if (state.joint_names.size() < 2) throw std::invalid_argument("expected at least two joints");
    rll_policy::MotorCommand out; out.timestamp_ns=stamp; out.joint_names=state.joint_names;
    for (std::size_t i=0;i<state.joint_names.size();++i) {
      out.position.push_back(values[static_cast<std::int64_t>(i%2)].item<double>());
      out.velocity.push_back(0.0); out.kp.push_back(10.0+static_cast<double>(i));
      out.kd.push_back(0.5+static_cast<double>(i)*0.1); out.torque.push_back(0.0);
    }
    return out;
  }
  void publish(const rll_policy::MotorCommand& command) override { if (writer_) writer_(context_, &command); }
 private:
  rll_policy::MotorCommandWriteFn writer_{}; void* context_{};
};
}

extern "C" {
std::uint32_t rll_robot_plugin_abi_version() { return RLL_PLUGIN_ABI; }
rll_policy::InputProcessor* rll_create_input_processor_v1(const char*) { return new TestInput(); }
void rll_destroy_input_processor_v1(rll_policy::InputProcessor* value) { delete value; }
rll_policy::OutputProcessor* rll_create_output_processor_v1(const char*, rll_policy::MotorCommandWriteFn writer, void* context) {
  return new TestOutput(writer, context);
}
void rll_destroy_output_processor_v1(rll_policy::OutputProcessor* value) { delete value; }
}
