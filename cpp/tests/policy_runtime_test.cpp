#include "rll_policy/config.hpp"
#include "rll_policy/runtime_utils.hpp"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iterator>
#include <filesystem>
#include <stdexcept>
#include <string>

using namespace std::chrono_literals;

namespace {
rll_policy::MotorCommand command(std::vector<std::string> names, std::vector<double> position,
                                 std::vector<double> kp, std::vector<double> kd) {
  rll_policy::MotorCommand result; result.control_mode=rll_policy::MotorControlMode::mit;
  result.joint_names=std::move(names); result.position=std::move(position);
  result.velocity={0.0,0.0}; result.kp=std::move(kp); result.kd=std::move(kd); result.torque={0.0,0.0};
  return result;
}
template <typename F> bool rejects(F&& f) { try { f(); } catch (const std::exception&) { return true; } return false; }
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const auto config=rll_policy::PolicyRuntimeConfig::load(argv[1]);
  assert(config.transition_duration_s == 0.5);
  assert(config.policies.size() == 2 && config.policies[0].index == 1);
  rll_policy::RuntimeStateMachine fsm(config.state_machine);
  assert(fsm.state() == "damping");
  assert(fsm.on_mode(1) && fsm.state() == "velocity");
  assert(!fsm.on_mode(9) && fsm.state() == "velocity");
  assert(fsm.on_mode(2) && fsm.state() == "tracking");
  assert(fsm.on_mode(0) && fsm.state() == "damping");
  assert(fsm.on_button(0, 2) && fsm.state() == "velocity");

  rll_policy::RobotState state; state.joint_names={"hip","ankle"};
  auto damping=rll_policy::make_damping_command(state, 0.7, 123);
  assert(damping.timestamp_ns == 123 && damping.kp == std::vector<double>({0,0}));
  assert(damping.kd == std::vector<double>({0.7,0.7}));
  assert(damping.position == std::vector<double>({0,0}));

  auto left=command({"hip","ankle"},{0.0,10.0},{2.0,4.0},{1.0,3.0});
  auto right=command({"ankle","hip"},{20.0,2.0},{8.0,6.0},{7.0,5.0});
  auto mixed=rll_policy::blend_motor_commands(left,right,0.5,456);
  assert(mixed.timestamp_ns == 456 && mixed.joint_names == left.joint_names);
  assert(mixed.position == std::vector<double>({1.0,15.0}));
  assert(mixed.kp == std::vector<double>({4.0,6.0}) && mixed.kd == std::vector<double>({3.0,5.0}));
  auto damping_mix=rll_policy::blend_motor_commands(left,damping,0.5,456);
  assert(damping_mix.kp == std::vector<double>({1.0,2.0}));

  auto duplicate=right; duplicate.joint_names={"hip","hip"};
  auto missing=right; missing.joint_names={"hip","knee"};
  auto bad_size=right; bad_size.kp={1.0};
  assert(rejects([&]{ (void)rll_policy::blend_motor_commands(left,duplicate,0.5,0); }));
  assert(rejects([&]{ (void)rll_policy::blend_motor_commands(left,missing,0.5,0); }));
  assert(rejects([&]{ (void)rll_policy::blend_motor_commands(left,bad_size,0.5,0); }));
  assert(rll_policy::inputs_fresh(10.0,9.9,9.8,0.2,0.3,true,true));
  assert(!rll_policy::inputs_fresh(10.0,9.0,9.9,0.2,0.3,true,true));
  assert(!rll_policy::inputs_fresh(10.0,9.9,9.9,0.2,0.3,false,true));
  assert(!rll_policy::active_inputs_fresh(false,10.0,9.9,9.9,0.2,0.3,true,true));
  assert(rll_policy::active_inputs_fresh(true,10.0,9.9,9.9,0.2,0.3,true,true));

  std::ifstream source(argv[1]);
  std::string invalid_yaml((std::istreambuf_iterator<char>(source)),std::istreambuf_iterator<char>());
  auto second_index=invalid_yaml.find("index: 2");
  assert(second_index!=std::string::npos);
  invalid_yaml.replace(second_index,std::string("index: 2").size(),"index: 1");
  auto invalid_path=std::filesystem::temp_directory_path()/"rll-policy-duplicate-index.yaml";
  { std::ofstream invalid_file(invalid_path); invalid_file << invalid_yaml; }
  assert(rejects([&]{ (void)rll_policy::PolicyRuntimeConfig::load(invalid_path); }));
  std::filesystem::remove(invalid_path);

  auto t0=std::chrono::steady_clock::time_point{};
  rll_policy::OutputTransition transition;
  transition.begin(damping,left,t0,500ms);
  auto mid=transition.sample(t0+250ms,1);
  assert(mid.position == std::vector<double>({0.0,5.0}));
  auto next=command({"hip","ankle"},{4.0,6.0},{8.0,8.0},{2.0,2.0});
  transition.begin(mid,next,t0+250ms,500ms);
  auto restarted=transition.sample(t0+250ms,2);
  assert(restarted.position == mid.position && restarted.kp == mid.kp);
  assert(transition.sample(t0+750ms,3).position == next.position);
}
