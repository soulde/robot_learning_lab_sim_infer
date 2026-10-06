#include "rll_policy/detail/dds_messages.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace rll_policy::detail {
namespace {
template <std::size_t N, typename T>
void check_size(const std::vector<T>& values, const char* name) {
  if (values.size() > N) throw std::invalid_argument(std::string(name) + " exceeds IDL bound");
}
template <std::size_t N>
void check_names(const std::vector<std::string>& names) {
  check_size<N>(names, "joint_names");
  for (const auto& name : names)
    if (name.size() > 64) throw std::invalid_argument("joint name exceeds IDL bound of 64 characters");
}
void set_double_seq(dds_sequence_double& out, const std::vector<double>& in) {
  out._maximum = static_cast<std::uint32_t>(in.size()); out._length = out._maximum;
  out._buffer = in.empty() ? nullptr : dds_sequence_double_allocbuf(in.size());
  if (!in.empty() && !out._buffer) throw std::bad_alloc();
  out._release = true;
  std::copy(in.begin(), in.end(), out._buffer);
}
void set_names(dds_sequence_string64& out, const std::vector<std::string>& in) {
  out._maximum = static_cast<std::uint32_t>(in.size()); out._length = out._maximum;
  out._buffer = in.empty() ? nullptr : dds_sequence_string64_allocbuf(in.size());
  if (!in.empty() && !out._buffer) throw std::bad_alloc();
  out._release = true;
  for (std::size_t i = 0; i < in.size(); ++i) std::memcpy(out._buffer[i], in[i].c_str(), in[i].size() + 1);
}
std::vector<double> read_seq(const dds_sequence_double& seq) {
  if (seq._length && !seq._buffer) throw std::invalid_argument("IDL sequence has null buffer");
  if (!seq._length) return {};
  return {seq._buffer, seq._buffer + seq._length};
}
std::vector<std::string> read_names(const dds_sequence_string64& seq) {
  if (seq._length && !seq._buffer) throw std::invalid_argument("IDL sequence has null buffer");
  std::vector<std::string> names; names.reserve(seq._length);
  for (std::uint32_t i = 0; i < seq._length; ++i) names.emplace_back(seq._buffer[i]);
  return names;
}
template <std::size_t N> void read_array(std::array<double, N>& out, const double (&in)[N]) {
  std::copy(std::begin(in), std::end(in), out.begin());
}
template <std::size_t N> void write_array(double (&out)[N], const std::array<double, N>& in) {
  std::copy(in.begin(), in.end(), std::begin(out));
}
}

robot_learning_lab_sim_infer_msg_v1_RobotState to_idl(const RobotState& v) {
  check_names<64>(v.joint_names); check_size<64>(v.joint_position, "joint_position");
  check_size<64>(v.joint_velocity, "joint_velocity"); check_size<64>(v.joint_effort, "joint_effort");
  check_size<256>(v.contact_force_xyz, "contact_force_xyz");
  robot_learning_lab_sim_infer_msg_v1_RobotState out{};
  out.timestamp_ns = v.timestamp_ns; set_names(out.joint_names, v.joint_names);
  set_double_seq(out.joint_position, v.joint_position);
  set_double_seq(out.joint_velocity, v.joint_velocity);
  set_double_seq(out.joint_effort, v.joint_effort);
  write_array(out.base_position, v.base_position); write_array(out.base_orientation_xyzw, v.base_orientation_xyzw);
  write_array(out.base_linear_velocity, v.base_linear_velocity); write_array(out.base_angular_velocity, v.base_angular_velocity);
  write_array(out.imu_orientation_xyzw, v.imu_orientation_xyzw); write_array(out.imu_angular_velocity, v.imu_angular_velocity);
  write_array(out.imu_linear_acceleration, v.imu_linear_acceleration); set_double_seq(out.contact_force_xyz, v.contact_force_xyz);
  return out;
}
RobotState from_idl(const robot_learning_lab_sim_infer_msg_v1_RobotState& v) {
  if (v.joint_names._length > 64 || v.joint_position._length > 64 || v.joint_velocity._length > 64 ||
      v.joint_effort._length > 64 || v.contact_force_xyz._length > 256) throw std::invalid_argument("RobotState exceeds IDL bounds");
  RobotState out; out.timestamp_ns = v.timestamp_ns; out.joint_names = read_names(v.joint_names);
  out.joint_position = read_seq(v.joint_position); out.joint_velocity = read_seq(v.joint_velocity); out.joint_effort = read_seq(v.joint_effort);
  read_array(out.base_position, v.base_position); read_array(out.base_orientation_xyzw, v.base_orientation_xyzw);
  read_array(out.base_linear_velocity, v.base_linear_velocity); read_array(out.base_angular_velocity, v.base_angular_velocity);
  read_array(out.imu_orientation_xyzw, v.imu_orientation_xyzw); read_array(out.imu_angular_velocity, v.imu_angular_velocity);
  read_array(out.imu_linear_acceleration, v.imu_linear_acceleration); out.contact_force_xyz = read_seq(v.contact_force_xyz); return out;
}
robot_learning_lab_sim_infer_msg_v1_RCCommand to_idl(const RCCommand& v) {
  robot_learning_lab_sim_infer_msg_v1_RCCommand out{}; out.timestamp_ns=v.timestamp_ns; out.enabled=v.enabled; out.mode=v.mode;
  out.vx=v.vx; out.vy=v.vy; out.yaw_rate=v.yaw_rate; out.body_height=v.body_height; out.button_mask=v.button_mask; return out;
}
RCCommand from_idl(const robot_learning_lab_sim_infer_msg_v1_RCCommand& v) {
  return {v.timestamp_ns,v.enabled,v.mode,v.vx,v.vy,v.yaw_rate,v.body_height,v.button_mask};
}
robot_learning_lab_sim_infer_msg_v1_MotorCommand to_idl(const MotorCommand& v) {
  check_names<64>(v.joint_names); check_size<64>(v.position,"position"); check_size<64>(v.velocity,"velocity");
  check_size<64>(v.kp,"kp"); check_size<64>(v.kd,"kd"); check_size<64>(v.torque,"torque");
  robot_learning_lab_sim_infer_msg_v1_MotorCommand out{};
  out.timestamp_ns=v.timestamp_ns; out.control_mode=static_cast<robot_learning_lab_sim_infer_msg_v1_MotorControlMode>(v.control_mode);
  set_names(out.joint_names,v.joint_names); set_double_seq(out.position,v.position); set_double_seq(out.velocity,v.velocity);
  set_double_seq(out.kp,v.kp); set_double_seq(out.kd,v.kd); set_double_seq(out.torque,v.torque); return out;
}
MotorCommand from_idl(const robot_learning_lab_sim_infer_msg_v1_MotorCommand& v) {
  if (v.control_mode < robot_learning_lab_sim_infer_msg_v1_POSITION || v.control_mode > robot_learning_lab_sim_infer_msg_v1_MIT) throw std::invalid_argument("invalid MotorControlMode");
  MotorCommand out; out.timestamp_ns=v.timestamp_ns; out.control_mode=static_cast<MotorControlMode>(v.control_mode);
  out.joint_names=read_names(v.joint_names); out.position=read_seq(v.position); out.velocity=read_seq(v.velocity);
  out.kp=read_seq(v.kp); out.kd=read_seq(v.kd); out.torque=read_seq(v.torque); return out;
}
}
