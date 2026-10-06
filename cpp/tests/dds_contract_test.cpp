#include "rll_policy/detail/dds_messages.hpp"

#include <cassert>
#include <stdexcept>

int main() {
  using namespace rll_policy;
  using namespace rll_policy::detail;
  static_assert(static_cast<int>(MotorControlMode::position) == 0);
  static_assert(static_cast<int>(MotorControlMode::mit) == 3);

  RobotState state;
  state.timestamp_ns = 42;
  state.joint_names = {"left", "right"};
  state.joint_position = {0.1, -0.2};
  state.joint_velocity = {1.0, -1.0};
  state.joint_effort = {3.0, 4.0};
  state.contact_force_xyz = {1.0, 2.0, 3.0};
  auto state_idl = to_idl(state);
  auto state_roundtrip = from_idl(state_idl);
  assert(state_roundtrip.timestamp_ns == state.timestamp_ns);
  assert(state_roundtrip.joint_names == state.joint_names);
  assert(state_roundtrip.joint_position == state.joint_position);
  assert(state_roundtrip.contact_force_xyz == state.contact_force_xyz);
  robot_learning_lab_sim_infer_msg_v1_RobotState_free(&state_idl, DDS_FREE_CONTENTS);

  RCCommand rc;
  rc.timestamp_ns = 99;
  rc.enabled = true;
  rc.mode = 4;
  rc.vx = 0.75F;
  auto rc_roundtrip = from_idl(to_idl(rc));
  assert(rc_roundtrip.timestamp_ns == 99);
  assert(rc_roundtrip.enabled && rc_roundtrip.mode == 4);
  assert(rc_roundtrip.vx == 0.75F);

  MotorCommand motor;
  motor.control_mode = MotorControlMode::mit;
  motor.joint_names = {"left"};
  motor.position = {0.2};
  motor.velocity = {0.3};
  motor.kp = {20.0};
  motor.kd = {0.8};
  motor.torque = {1.2};
  auto motor_idl = to_idl(motor);
  auto motor_roundtrip = from_idl(motor_idl);
  assert(motor_roundtrip.control_mode == MotorControlMode::mit);
  assert(motor_roundtrip.joint_names == motor.joint_names);
  assert(motor_roundtrip.kp == motor.kp && motor_roundtrip.torque == motor.torque);
  robot_learning_lab_sim_infer_msg_v1_MotorCommand_free(&motor_idl, DDS_FREE_CONTENTS);

  state.joint_names.resize(65);
  bool rejected = false;
  try {
    (void)to_idl(state);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}
