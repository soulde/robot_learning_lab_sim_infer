#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rll_policy {

enum class MotorControlMode : std::uint32_t { position = 0, velocity = 1, torque = 2, mit = 3 };

struct RobotState {
  std::uint64_t timestamp_ns{};
  std::vector<std::string> joint_names;
  std::vector<double> joint_position, joint_velocity, joint_effort;
  std::array<double, 3> base_position{};
  std::array<double, 4> base_orientation_xyzw{};
  std::array<double, 3> base_linear_velocity{}, base_angular_velocity{};
  std::array<double, 4> imu_orientation_xyzw{};
  std::array<double, 3> imu_angular_velocity{}, imu_linear_acceleration{};
  std::vector<double> contact_force_xyz;
};

struct RCCommand {
  std::uint64_t timestamp_ns{};
  bool enabled{};
  std::uint16_t mode{};
  float vx{}, vy{}, yaw_rate{};
  std::uint32_t button_mask{};
};

struct MotorCommand {
  std::uint64_t timestamp_ns{};
  MotorControlMode control_mode{MotorControlMode::mit};
  std::vector<std::string> joint_names;
  std::vector<double> position, velocity, kp, kd, torque;
};

}  // namespace rll_policy
