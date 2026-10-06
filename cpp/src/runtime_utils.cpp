#include "rll_policy/runtime_utils.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace rll_policy {
namespace {
void validate(const MotorCommand& command) {
  const auto n=command.joint_names.size();
  if (n==0 || n>64) throw std::invalid_argument("MotorCommand must contain 1 to 64 joints");
  if (std::set<std::string>(command.joint_names.begin(),command.joint_names.end()).size()!=n)
    throw std::invalid_argument("MotorCommand joint names contain duplicates");
  const auto check=[n](const std::vector<double>& values,const char* name) {
    if (values.size()!=n) throw std::invalid_argument(std::string("MotorCommand ")+name+" length does not match joint names");
    if (!std::all_of(values.begin(),values.end(),[](double v){return std::isfinite(v);}))
      throw std::invalid_argument(std::string("MotorCommand ")+name+" contains NaN or Inf");
  };
  check(command.position,"position"); check(command.velocity,"velocity"); check(command.kp,"kp"); check(command.kd,"kd"); check(command.torque,"torque");
}
}
bool inputs_fresh(double now,double state_time,double rc_time,double state_timeout,double rc_timeout,bool has_state,bool has_rc) {
  return has_state && has_rc && std::isfinite(now) && now>=state_time && now>=rc_time &&
         now-state_time<=state_timeout && now-rc_time<=rc_timeout;
}
bool active_inputs_fresh(bool enabled,double now,double state_time,double rc_time,double state_timeout,double rc_timeout,bool has_state,bool has_rc) {
  return enabled && inputs_fresh(now,state_time,rc_time,state_timeout,rc_timeout,has_state,has_rc);
}
MotorCommand make_damping_command(const RobotState& state,double kd,std::uint64_t stamp) {
  if (state.joint_names.empty() || state.joint_names.size()>64) throw std::invalid_argument("RobotState needs 1 to 64 joints for damping");
  if (std::set<std::string>(state.joint_names.begin(),state.joint_names.end()).size()!=state.joint_names.size()) throw std::invalid_argument("RobotState joint names contain duplicates");
  if (!std::isfinite(kd) || kd<0) throw std::invalid_argument("damping kd must be finite and non-negative");
  const auto count=state.joint_names.size();
  MotorCommand out; out.timestamp_ns=stamp; out.control_mode=MotorControlMode::mit; out.joint_names=state.joint_names;
  out.position.assign(count,0.0); out.velocity.assign(count,0.0); out.kp.assign(count,0.0); out.kd.assign(count,kd); out.torque.assign(count,0.0);
  return out;
}
MotorCommand blend_motor_commands(const MotorCommand& from,const MotorCommand& to,double alpha,std::uint64_t stamp) {
  validate(from); validate(to);
  if (!std::isfinite(alpha) || alpha<0.0 || alpha>1.0) throw std::invalid_argument("blend alpha must be in [0,1]");
  if (from.control_mode!=to.control_mode) throw std::invalid_argument("cannot blend different motor control modes");
  if (from.joint_names.size()!=to.joint_names.size()) throw std::invalid_argument("MotorCommand joint sets differ");
  std::unordered_map<std::string,std::size_t> lookup;
  for (std::size_t i=0;i<to.joint_names.size();++i) lookup.emplace(to.joint_names[i],i);
  for (const auto& name:from.joint_names) if (!lookup.count(name)) throw std::invalid_argument("MotorCommand joint sets differ: missing "+name);
  MotorCommand out; out.timestamp_ns=stamp; out.control_mode=to.control_mode; out.joint_names=from.joint_names;
  const auto mix=[&](const std::vector<double>& a,const std::vector<double>& b) {
    std::vector<double> values; values.reserve(a.size());
    for (std::size_t i=0;i<a.size();++i) values.push_back((1.0-alpha)*a[i]+alpha*b[lookup.at(from.joint_names[i])]);
    return values;
  };
  out.position=mix(from.position,to.position); out.velocity=mix(from.velocity,to.velocity); out.kp=mix(from.kp,to.kp); out.kd=mix(from.kd,to.kd); out.torque=mix(from.torque,to.torque);
  validate(out); return out;
}
void OutputTransition::begin(const MotorCommand& from,const MotorCommand& to,Clock::time_point start,std::chrono::duration<double> duration) {
  if (!std::isfinite(duration.count()) || duration.count()<0) throw std::invalid_argument("transition duration must be finite and non-negative");
  (void)blend_motor_commands(from,to,0.0,0);
  from_=from; to_=to; start_=start; duration_=duration; active_=true;
}
MotorCommand OutputTransition::sample(Clock::time_point now,std::uint64_t stamp) const {
  if (!active_) throw std::logic_error("transition has not started");
  double progress=duration_.count()==0 ? 1.0 : std::chrono::duration<double>(now-start_).count()/duration_.count();
  progress=std::clamp(progress,0.0,1.0); const double alpha=progress*progress*(3.0-2.0*progress);
  return blend_motor_commands(from_,to_,alpha,stamp);
}
bool OutputTransition::complete(Clock::time_point now) const noexcept { return active_ && now>=start_+duration_; }
}
