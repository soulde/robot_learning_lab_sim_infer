#include "rll_policy/plugin_loader.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {
std::vector<rll_policy::MotorCommand> published;
void capture(void*, const rll_policy::MotorCommand* command) { published.push_back(*command); }
}

int main(int argc, char** argv) {
  assert(argc == 5);
  auto plugin = rll_policy::RobotPluginHandle::load(argv[1], "test-config.yaml", capture, nullptr);
  rll_policy::RobotState state;
  state.joint_names = {"joint_a", "joint_b"};
  state.joint_position = {1.25, -0.5};
  rll_policy::RCCommand rc;
  rc.vx = 0.75F;
  auto obs = plugin.input().build_observation("walk", state, rc);
  assert(obs.sizes() == torch::IntArrayRef({1, 2}));
  assert(obs[0][0].item<float>() == 1.25F && obs[0][1].item<float>() == 0.75F);

  auto action = torch::tensor({2.0F, -3.0F});
  auto command = plugin.output().make_command("walk", action, state, 1234);
  assert(command.timestamp_ns == 1234 && command.joint_names == state.joint_names);
  assert(command.position == std::vector<double>({2.0, -3.0}));
  assert(command.kp == std::vector<double>({10.0, 11.0}));
  plugin.output().publish(command);
  assert(published.size() == 1 && published[0].position == command.position);

  for (int i = 2; i < 5; ++i) {
    bool rejected = false;
    try { (void)rll_policy::RobotPluginHandle::load(argv[i], "test-config.yaml", capture, nullptr); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
}
