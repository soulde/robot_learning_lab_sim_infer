#pragma once

#include "rll_policy/config.hpp"
#include "rll_policy/messages.hpp"

#include <optional>

namespace rll_policy {

class DdsTransport {
 public:
  explicit DdsTransport(const DdsRuntimeConfig& config);
  DdsTransport(const DdsTransport&) = delete;
  DdsTransport& operator=(const DdsTransport&) = delete;
  ~DdsTransport();

  std::optional<RobotState> take_latest_state();
  std::optional<RCCommand> take_latest_rc();
  void publish_motor_command(const MotorCommand& command);
  void close() noexcept;

 private:
  int participant_{-1};
  int state_reader_{-1};
  int rc_reader_{-1};
  int motor_writer_{-1};
  int state_topic_{-1};
  int rc_topic_{-1};
  int motor_topic_{-1};
};

}  // namespace rll_policy
