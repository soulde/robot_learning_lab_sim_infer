#pragma once

#include "rll_policy/messages.hpp"

#include <chrono>

namespace rll_policy {

bool inputs_fresh(double now_s, double state_received_s, double rc_received_s,
                  double state_timeout_s, double rc_timeout_s,
                  bool has_state, bool has_rc);
bool active_inputs_fresh(bool enabled, double now_s, double state_received_s, double rc_received_s,
                         double state_timeout_s, double rc_timeout_s, bool has_state, bool has_rc);
MotorCommand make_damping_command(const RobotState& state, double kd, std::uint64_t timestamp_ns);
MotorCommand blend_motor_commands(const MotorCommand& from, const MotorCommand& to,
                                  double alpha, std::uint64_t timestamp_ns);

class OutputTransition {
 public:
  using Clock = std::chrono::steady_clock;
  void begin(const MotorCommand& from, const MotorCommand& to, Clock::time_point start,
             std::chrono::duration<double> duration);
  MotorCommand sample(Clock::time_point now, std::uint64_t timestamp_ns) const;
  bool complete(Clock::time_point now) const noexcept;
  bool active() const noexcept { return active_; }
 private:
  MotorCommand from_, to_;
  Clock::time_point start_{};
  std::chrono::duration<double> duration_{};
  bool active_{};
};

}  // namespace rll_policy
