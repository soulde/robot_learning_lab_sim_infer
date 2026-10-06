#include "rll_policy/plugin_api.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using Names = std::array<const char*, 23>;
constexpr Names kJointNames={
  "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
  "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
  "waist_pitch_joint","waist_roll_joint","waist_yaw_joint",
  "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
  "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"};
constexpr Names kTrackingNames={
  "waist_pitch_joint","left_shoulder_pitch_joint","right_shoulder_pitch_joint","waist_roll_joint","left_shoulder_roll_joint","right_shoulder_roll_joint",
  "waist_yaw_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_hip_pitch_joint","right_hip_pitch_joint","left_elbow_joint","right_elbow_joint",
  "left_hip_roll_joint","right_hip_roll_joint","left_hip_yaw_joint","right_hip_yaw_joint","left_knee_joint","right_knee_joint",
  "left_ankle_pitch_joint","right_ankle_pitch_joint","left_ankle_roll_joint","right_ankle_roll_joint"};
constexpr std::size_t kJoints=23;
struct Vec3 { double x{},y{},z{}; };
struct Quat { double x{},y{},z{},w{1.0}; };
struct MotionFrame { std::array<float,kJoints> position{},velocity{}; Vec3 torso_position; Quat torso_orientation; };
struct Motion { double fps{}; std::vector<MotionFrame> frames; };
std::size_t name_index(const Names& names,const std::string& name);
struct Context {
  std::array<double,kJoints> velocity_history{},tracking_history{};
  std::array<double,kJoints> default_position{};
  Motion motion;
  std::uint64_t tracking_start_ns{};
  bool tracking_started{};
  bool tracking_restart_pending{true};
  bool tracking_playing{};
  std::uint32_t previous_button_mask{};
  std::uint32_t motion_start_button_mask{1};
  Quat alignment_rotation{};
  Vec3 alignment_translation{};
};

std::array<double,kJoints> read_default_joint(const YAML::Node& config) {
  const auto node=config["default_joint"];
  if (!node || !node.IsMap()) throw std::invalid_argument("Chocolate plugin config requires default_joint mapping");
  if (node.size()!=kJoints) throw std::invalid_argument("default_joint must define exactly all 23 Chocolate joints");
  std::array<double,kJoints> values{};
  std::array<bool,kJoints> found{};
  for (const auto& entry:node) {
    if (!entry.first.IsScalar()) throw std::invalid_argument("default_joint keys must be joint names");
    const auto name=entry.first.as<std::string>();
    const auto index=name_index(kJointNames,name);
    const double value=entry.second.as<double>();
    if (!std::isfinite(value)) throw std::invalid_argument("default_joint values must be finite: "+name);
    if (found[index]) throw std::invalid_argument("duplicate default_joint entry: "+name);
    found[index]=true; values[index]=value;
  }
  for (std::size_t i=0;i<kJoints;++i)
    if (!found[i]) throw std::invalid_argument("default_joint is missing Chocolate joint: "+std::string(kJointNames[i]));
  return values;
}
std::array<double,kJoints> gains_kp() {
  std::array<double,kJoints> v{};
  for (std::size_t i=0;i<kJoints;++i) {
    const std::string n=kJointNames[i];
    v[i]=n.find("shoulder")!=std::string::npos || n.find("elbow")!=std::string::npos ? 10.0 :
         n.find("waist")!=std::string::npos ? 20.0 : n.find("ankle")!=std::string::npos ? 20.0 : 40.0;
  }
  return v;
}
std::array<double,kJoints> gains_kd() {
  std::array<double,kJoints> v{};
  for (std::size_t i=0;i<kJoints;++i) v[i]=(std::string(kJointNames[i]).find("waist_pitch")!=std::string::npos || std::string(kJointNames[i]).find("waist_roll")!=std::string::npos) ? 2.0 : 1.0;
  return v;
}
std::size_t name_index(const Names& names,const std::string& name) {
  for (std::size_t i=0;i<kJoints;++i) if (name==names[i]) return i;
  throw std::invalid_argument("unknown Chocolate joint name: "+name);
}
std::array<double,kJoints> ordered_values(const std::vector<std::string>& names,const std::vector<double>& values) {
  if (names.size()!=values.size() || names.size()!=kJoints) throw std::invalid_argument("RobotState must contain all 23 Chocolate joints");
  std::unordered_map<std::string,double> by_name;
  for (std::size_t i=0;i<names.size();++i) if (!by_name.emplace(names[i],values[i]).second) throw std::invalid_argument("RobotState contains duplicate joint name: "+names[i]);
  std::array<double,kJoints> result{};
  for (std::size_t i=0;i<kJoints;++i) {
    auto found=by_name.find(kJointNames[i]); if (found==by_name.end()) throw std::invalid_argument("RobotState is missing Chocolate joint: "+std::string(kJointNames[i]));
    result[i]=found->second;
  }
  return result;
}
Vec3 add(Vec3 a,Vec3 b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
Vec3 sub(Vec3 a,Vec3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
Vec3 rotate(Quat q,Vec3 v) {
  const Vec3 u{q.x,q.y,q.z};
  const Vec3 uv{u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x};
  const Vec3 uuv{u.y*uv.z-u.z*uv.y,u.z*uv.x-u.x*uv.z,u.x*uv.y-u.y*uv.x};
  return add(v,add({2.0*q.w*uv.x,2.0*q.w*uv.y,2.0*q.w*uv.z},{2.0*uuv.x,2.0*uuv.y,2.0*uuv.z}));
}
Quat multiply(Quat a,Quat b) { return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z}; }
Quat inverse(Quat q) { return {-q.x,-q.y,-q.z,q.w}; }
std::array<double,6> first_two_columns(Quat q) {
  const double xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z, xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z, wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
  return {1-2*(yy+zz),2*(xy-wz),2*(xy+wz),1-2*(xx+zz),2*(xz-wy),2*(yz+wx)};
}
std::filesystem::path motion_path(const YAML::Node& config, const char* config_path) {
  if (!config["motion_file"]) throw std::invalid_argument("Chocolate plugin config requires motion_file");
  std::string motion=config["motion_file"].as<std::string>();
  if (motion.rfind("~/",0)==0) {
    const char* home=std::getenv("HOME");
    if (!home) throw std::invalid_argument("cannot expand motion_file '~': HOME is not set");
    motion=std::string(home)+motion.substr(1);
  }
  std::filesystem::path p(motion);
  if (p.is_relative()) p=std::filesystem::absolute(std::filesystem::path(config_path).parent_path()/p);
  return p.lexically_normal();
}
template <typename T> void read_exact(std::ifstream& stream,T& value,const char* field) {
  if (!stream.read(reinterpret_cast<char*>(&value),sizeof(T))) throw std::runtime_error(std::string("truncated Chocolate motion file at ")+field);
}
Motion load_motion(const std::filesystem::path& path) {
  std::ifstream in(path,std::ios::binary);
  if (!in) throw std::runtime_error("cannot open Chocolate motion file: "+path.string());
  char magic[8]{}; if (!in.read(magic,sizeof(magic)) || std::string(magic,sizeof(magic))!="RLLCHOC1") throw std::runtime_error("invalid Chocolate motion file magic/version");
  std::uint32_t frames=0,joints=0; double fps=0;
  read_exact(in,frames,"frame count"); read_exact(in,joints,"joint count"); read_exact(in,fps,"FPS");
  if (frames==0 || joints!=kJoints || !std::isfinite(fps) || fps<=0) throw std::runtime_error("invalid Chocolate motion header");
  Motion result; result.fps=fps; result.frames.resize(frames);
  for (auto& frame:result.frames) {
    if (!in.read(reinterpret_cast<char*>(frame.position.data()),sizeof(float)*kJoints) ||
        !in.read(reinterpret_cast<char*>(frame.velocity.data()),sizeof(float)*kJoints)) throw std::runtime_error("truncated Chocolate motion joint arrays");
    float p[3],q[4];
    if (!in.read(reinterpret_cast<char*>(p),sizeof(p)) || !in.read(reinterpret_cast<char*>(q),sizeof(q))) throw std::runtime_error("truncated Chocolate motion torso arrays");
    frame.torso_position={p[0],p[1],p[2]}; frame.torso_orientation={q[0],q[1],q[2],q[3]};
  }
  char trailing; if (in.read(&trailing,1)) throw std::runtime_error("Chocolate motion file has unexpected trailing bytes");
  return result;
}
std::shared_ptr<Context> shared_context(const char* path) {
  static std::weak_ptr<Context> weak;
  auto context=weak.lock();
  if (!context) {
    const auto config=YAML::LoadFile(path);
    context=std::make_shared<Context>();
    context->default_position=read_default_joint(config);
    if (config["motion_start_button_mask"])
      context->motion_start_button_mask=config["motion_start_button_mask"].as<std::uint32_t>();
    if (!context->motion_start_button_mask ||
        (context->motion_start_button_mask & (context->motion_start_button_mask-1)))
      throw std::invalid_argument("motion_start_button_mask must be a single-bit mask");
    context->motion=load_motion(motion_path(config,path));
    const double expected=config["expected_fps"] ? config["expected_fps"].as<double>() : 50.0;
    if (!std::isfinite(expected) || std::abs(context->motion.fps-expected)>1e-6)
      throw std::runtime_error("Chocolate motion FPS does not match configured expected_fps");
    weak=context;
  }
  return context;
}
class ChocolateInput final : public rll_policy::InputProcessor {
 public:
  explicit ChocolateInput(std::shared_ptr<Context> context):context_(std::move(context)) {}
  void on_policy_selected(std::string_view type) override {
    // Each activation starts a fresh actor history. In particular, returning
    // from the fixed-pose state must not feed actions from an earlier velocity
    // run (or a simulator reset) into the policy's previous-action observation.
    context_->velocity_history.fill(0.0);
    context_->tracking_history.fill(0.0);
    context_->tracking_started=false;
    context_->tracking_restart_pending=true;
    context_->tracking_playing=false;
    context_->previous_button_mask=0;
    if (type!="tracking") context_->tracking_start_ns=0;
  }
  torch::Tensor build_observation(std::string_view type,const rll_policy::RobotState& state,const rll_policy::RCCommand& rc) override {
    if (type=="velocity") return velocity(state,rc);
    if (type=="tracking") return tracking(state,rc);
    throw std::invalid_argument("unknown Chocolate policy type: "+std::string(type));
  }
 private:
  torch::Tensor velocity(const rll_policy::RobotState& state,const rll_policy::RCCommand& rc) {
    (void)rc;
    const auto position=ordered_values(state.joint_names,state.joint_position);
    const auto velocity=ordered_values(state.joint_names,state.joint_velocity);
    Quat q{state.base_orientation_xyzw[0],state.base_orientation_xyzw[1],state.base_orientation_xyzw[2],state.base_orientation_xyzw[3]};
    auto gravity=rotate(inverse(q),{0.0,0.0,-1.0});
    std::vector<float> obs; obs.reserve(78);
    for (double x:state.base_angular_velocity) obs.push_back(static_cast<float>(x*0.25));
    obs.insert(obs.end(),{static_cast<float>(gravity.x),static_cast<float>(gravity.y),static_cast<float>(gravity.z)});
    obs.insert(obs.end(),{rc.vx*2.0f,rc.vy*2.0f,rc.yaw_rate*0.25f});
    for (std::size_t i=0;i<kJoints;++i) obs.push_back(static_cast<float>(position[i]-context_->default_position[i]));
    for (double x:velocity) obs.push_back(static_cast<float>(x*0.05));
    for (double x:context_->velocity_history) obs.push_back(static_cast<float>(x));
    for (auto& x:obs) x=std::clamp(x,-100.0f,100.0f);
    return torch::from_blob(obs.data(),{1,78},torch::TensorOptions().dtype(torch::kFloat32)).clone();
  }
  torch::Tensor tracking(const rll_policy::RobotState& state,const rll_policy::RCCommand& rc) {
    const auto position=ordered_values(state.joint_names,state.joint_position);
    const auto velocity=ordered_values(state.joint_names,state.joint_velocity);
    auto& motion=context_->motion;
    if (!context_->tracking_started || context_->tracking_restart_pending) {
      context_->tracking_started=true; context_->tracking_start_ns=state.timestamp_ns;
      context_->tracking_playing=false;
      const auto& first=motion.frames.front();
      Quat robot{state.base_orientation_xyzw[0],state.base_orientation_xyzw[1],state.base_orientation_xyzw[2],state.base_orientation_xyzw[3]};
      context_->alignment_rotation=multiply(robot,inverse(first.torso_orientation));
      context_->alignment_translation=sub({state.base_position[0],state.base_position[1],state.base_position[2]},rotate(context_->alignment_rotation,first.torso_position));
      context_->tracking_restart_pending=false;
    }
    const auto pressed=rc.button_mask&~context_->previous_button_mask;
    context_->previous_button_mask=rc.button_mask;
    if (pressed&context_->motion_start_button_mask) {
      context_->tracking_playing=true;
      context_->tracking_start_ns=state.timestamp_ns;
    }
    const double elapsed=context_->tracking_playing && state.timestamp_ns>=context_->tracking_start_ns
        ? static_cast<double>(state.timestamp_ns-context_->tracking_start_ns)*1e-9 : 0.0;
    const auto frame_index=std::min<std::size_t>(static_cast<std::size_t>(std::llround(elapsed*motion.fps)),motion.frames.size()-1);
    const auto& reference=motion.frames[frame_index];
    Vec3 ref_position=add(rotate(context_->alignment_rotation,reference.torso_position),context_->alignment_translation);
    Quat ref_orientation=multiply(context_->alignment_rotation,reference.torso_orientation);
    Vec3 robot_position{state.base_position[0],state.base_position[1],state.base_position[2]};
    Quat robot_orientation{state.base_orientation_xyzw[0],state.base_orientation_xyzw[1],state.base_orientation_xyzw[2],state.base_orientation_xyzw[3]};
    Vec3 relative_position=rotate(inverse(robot_orientation),sub(ref_position,robot_position));
    Quat relative_orientation=multiply(inverse(robot_orientation),ref_orientation);
    const auto rotation=first_two_columns(relative_orientation);
    std::vector<float> obs; obs.reserve(124);
    for (float x:reference.position) obs.push_back(x);
    for (float x:reference.velocity) obs.push_back(x);
    obs.insert(obs.end(),{static_cast<float>(relative_position.x),static_cast<float>(relative_position.y),static_cast<float>(relative_position.z)});
    for (double x:rotation) obs.push_back(static_cast<float>(x));
    for (std::size_t i=0;i<kJoints;++i) {
      const auto primary_index=name_index(kJointNames,kTrackingNames[i]);
      obs.push_back(static_cast<float>(position[primary_index]-context_->default_position[primary_index]));
    }
    for (std::size_t i=0;i<kJoints;++i) obs.push_back(static_cast<float>(velocity[name_index(kJointNames,kTrackingNames[i])]));
    for (double x:context_->tracking_history) obs.push_back(static_cast<float>(x));
    return torch::from_blob(obs.data(),{1,124},torch::TensorOptions().dtype(torch::kFloat32)).clone();
  }
  std::shared_ptr<Context> context_;
};

class ChocolateOutput final : public rll_policy::OutputProcessor {
 public:
  ChocolateOutput(std::shared_ptr<Context> context,rll_policy::MotorCommandWriteFn writer,void* data)
      :context_(std::move(context)),writer_(writer),writer_context_(data),kp_(gains_kp()),kd_(gains_kd()) {}
  rll_policy::MotorCommand make_command(std::string_view policy,const torch::Tensor& action,const rll_policy::RobotState&,std::uint64_t stamp) override {
    auto values=action.to(torch::kCPU).to(torch::kFloat32).contiguous().view(-1);
    if (values.numel()!=static_cast<std::int64_t>(kJoints) || !torch::isfinite(values).all().item<bool>()) throw std::invalid_argument("Chocolate policy action must have 23 finite values");
    std::array<double,kJoints> primary_action{};
    if (policy=="velocity") {
      for (std::size_t i=0;i<kJoints;++i) { primary_action[i]=values[i].item<float>(); context_->velocity_history[i]=primary_action[i]; }
    } else if (policy=="tracking") {
      for (std::size_t i=0;i<kJoints;++i) {
        const double x=values[i].item<float>(); context_->tracking_history[i]=x;
        primary_action[name_index(kJointNames,kTrackingNames[i])]=x;
      }
    } else throw std::invalid_argument("unknown Chocolate policy: "+std::string(policy));
    rll_policy::MotorCommand out; out.timestamp_ns=stamp; out.control_mode=rll_policy::MotorControlMode::mit;
    for (std::size_t i=0;i<kJoints;++i) {
      out.joint_names.emplace_back(kJointNames[i]); out.position.push_back(context_->default_position[i]+0.5*primary_action[i]);
      out.velocity.push_back(0.0); out.kp.push_back(kp_[i]); out.kd.push_back(kd_[i]); out.torque.push_back(0.0);
    }
    return out;
  }
  rll_policy::MotorCommand make_fixed_pose_command(const rll_policy::RobotState&,std::uint64_t stamp) override {
    rll_policy::MotorCommand out; out.timestamp_ns=stamp; out.control_mode=rll_policy::MotorControlMode::mit;
    for (std::size_t i=0;i<kJoints;++i) {
      out.joint_names.emplace_back(kJointNames[i]); out.position.push_back(context_->default_position[i]);
      out.velocity.push_back(0.0); out.kp.push_back(kp_[i]); out.kd.push_back(kd_[i]); out.torque.push_back(0.0);
    }
    return out;
  }
  void publish(const rll_policy::MotorCommand& command) override { if (writer_) writer_(writer_context_,&command); }
 private:
  std::shared_ptr<Context> context_; rll_policy::MotorCommandWriteFn writer_{}; void* writer_context_{};
  std::array<double,kJoints> kp_,kd_;
};
}

extern "C" {
std::uint32_t rll_robot_plugin_abi_version() { return rll_policy::kRobotPluginAbiVersion; }
rll_policy::InputProcessor* rll_create_input_processor_v2(const char* config) { return new ChocolateInput(shared_context(config)); }
void rll_destroy_input_processor_v2(rll_policy::InputProcessor* value) { delete value; }
rll_policy::OutputProcessor* rll_create_output_processor_v2(const char* config,rll_policy::MotorCommandWriteFn writer,void* context) {
  return new ChocolateOutput(shared_context(config),writer,context);
}
void rll_destroy_output_processor_v2(rll_policy::OutputProcessor* value) { delete value; }
}
