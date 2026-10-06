#include "rll_policy/plugin_loader.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
void capture_command(void* context, const rll_policy::MotorCommand* command) {
  auto* count=static_cast<std::size_t*>(context);
  assert(command && command->joint_names.size()==23);
  ++*count;
}
void write_motion(const std::filesystem::path& path) {
  std::ofstream out(path, std::ios::binary);
  const char magic[8] = {'R','L','L','C','H','O','C','1'};
  out.write(magic, sizeof(magic));
  const std::uint32_t frames=2, joints=23;
  const double fps=50.0;
  out.write(reinterpret_cast<const char*>(&frames), sizeof(frames));
  out.write(reinterpret_cast<const char*>(&joints), sizeof(joints));
  out.write(reinterpret_cast<const char*>(&fps), sizeof(fps));
  for (std::uint32_t frame=0; frame<frames; ++frame) {
    std::vector<float> joint_pos(23), joint_vel(23, 0.1f);
    for (std::size_t i=0;i<joint_pos.size();++i) joint_pos[i]=static_cast<float>(i)+0.25f*frame;
    const float torso_pos[3]={0.0f,0.0f,0.8f};
    const float torso_quat_xyzw[4]={0.0f,0.0f,0.0f,1.0f};
    out.write(reinterpret_cast<const char*>(joint_pos.data()), joint_pos.size()*sizeof(float));
    out.write(reinterpret_cast<const char*>(joint_vel.data()), joint_vel.size()*sizeof(float));
    out.write(reinterpret_cast<const char*>(torso_pos), sizeof(torso_pos));
    out.write(reinterpret_cast<const char*>(torso_quat_xyzw), sizeof(torso_quat_xyzw));
  }
}
bool close(double a, double b) { return std::abs(a-b)<1e-5; }
}

int main(int argc, char** argv) {
  assert(argc==2);
  const auto dir=std::filesystem::temp_directory_path()/"rll-chocolate-plugin-test";
  std::filesystem::create_directories(dir);
  const auto motion=dir/"motion.rllchoc";
  const auto config=dir/"plugin.yaml";
  write_motion(motion);
  { std::ofstream yaml(config); yaml << "motion_file: motion.rllchoc\n"; }

  std::optional<rll_policy::RobotPluginHandle> plugin;
  std::size_t published=0;
  plugin.emplace(rll_policy::RobotPluginHandle::load(argv[1],config,capture_command,&published));
  rll_policy::RobotState state;
  state.timestamp_ns=1'000'000'000ULL;
  state.joint_names={
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
    "waist_pitch_joint","waist_roll_joint","waist_yaw_joint",
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"};
  state.joint_position.assign(23,0.0); state.joint_velocity.assign(23,0.0);
  state.base_position={0.0,0.0,0.8}; state.base_orientation_xyzw={0.0,0.0,0.0,1.0};
  state.base_angular_velocity={1.0,2.0,3.0};
  rll_policy::RCCommand rc; rc.enabled=true; rc.vx=0.5f; rc.vy=-0.25f; rc.yaw_rate=0.2f;

  auto velocity=plugin->input().build_observation("velocity",state,rc).to(torch::kCPU).contiguous().view(-1);
  assert(velocity.numel()==78);
  assert(close(velocity[0].item<double>(),0.25));
  assert(close(velocity[3].item<double>(),0.0) && close(velocity[5].item<double>(),-1.0));
  assert(close(velocity[6].item<double>(),1.0) && close(velocity[7].item<double>(),-0.5));
  assert(close(velocity[8].item<double>(),0.05));

  rc.mode=2;
  auto tracking=plugin->input().build_observation("tracking",state,rc).to(torch::kCPU).contiguous().view(-1);
  assert(tracking.numel()==124);
  assert(close(tracking[0].item<double>(),0.0));
  assert(close(tracking[1].item<double>(),1.0) && close(tracking[9].item<double>(),9.0));
  assert(close(tracking[23].item<double>(),0.1));
  assert(close(tracking[46].item<double>(),0.0) && close(tracking[48].item<double>(),0.0));
  assert(close(tracking[49].item<double>(),1.0));
  assert(close(tracking[64].item<double>(),0.312414));

  auto zero_action=torch::zeros({1,23},torch::kFloat32);
  auto command=plugin->output().make_command("tracking",zero_action,state,99);
  assert(command.joint_names.size()==23 && command.position.size()==23);
  assert(command.control_mode==rll_policy::MotorControlMode::mit);
  assert(close(command.position[11],-0.312414));
  assert(close(command.kp[11],40.0) && close(command.kd[11],1.0));
  auto unit_action=torch::ones({1,23},torch::kFloat32);
  command=plugin->output().make_command("tracking",unit_action,state,100);
  assert(close(command.position[11],-0.312414+0.5));
  plugin->output().publish(command);
  assert(published==1);
  std::vector<float> ordered_action(23);
  for (std::size_t i=0;i<ordered_action.size();++i) ordered_action[i]=static_cast<float>(i);
  command=plugin->output().make_command("tracking",torch::from_blob(ordered_action.data(),{1,23}).clone(),state,101);
  assert(close(command.position[11],-0.312414+0.5*9.0));
  auto history=plugin->input().build_observation("tracking",state,rc).to(torch::kCPU).contiguous().view(-1);
  assert(close(history[110].item<double>(),9.0));

  rc.mode=1;
  auto mode_one=plugin->input().build_observation("velocity",state,rc).to(torch::kCPU).contiguous().view(-1);
  assert(close(mode_one[55].item<double>(),0.0));
  state.timestamp_ns+=20'000'000ULL;
  rc.mode=2;
  auto restarted_tracking=plugin->input().build_observation("tracking",state,rc).to(torch::kCPU).contiguous().view(-1);
  assert(close(restarted_tracking[0].item<double>(),0.0)); // selecting tracking restarts the reference at frame 0
  assert(velocity.isfinite().all().item<bool>());
  auto permuted=state;
  std::reverse(permuted.joint_names.begin(),permuted.joint_names.end());
  std::reverse(permuted.joint_position.begin(),permuted.joint_position.end());
  std::reverse(permuted.joint_velocity.begin(),permuted.joint_velocity.end());
  auto reordered=plugin->input().build_observation("velocity",permuted,rc);
  auto canonical=plugin->input().build_observation("velocity",state,rc);
  assert(torch::allclose(reordered,canonical));
  state.joint_names[0]="unknown_joint";
  bool rejected=false;
  try { (void)plugin->input().build_observation("velocity",state,rc); }
  catch (const std::invalid_argument&) { rejected=true; }
  assert(rejected);
  plugin.reset();
  ::unlink(config.c_str());
  ::unlink(motion.c_str());
  ::rmdir(dir.c_str());
}
