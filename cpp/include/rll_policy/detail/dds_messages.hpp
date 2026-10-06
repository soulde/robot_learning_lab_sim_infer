#pragma once

#include "rll_policy/messages.hpp"
#include "robot_v1.h"

namespace rll_policy::detail {

// Returned IDL structs own sequence buffers. Release contents with the generated
// <Type>_free(&sample, DDS_FREE_CONTENTS) after a DDS write has copied the data.
robot_learning_lab_sim_infer_msg_v1_RobotState to_idl(const RobotState& value);
RobotState from_idl(const robot_learning_lab_sim_infer_msg_v1_RobotState& value);
robot_learning_lab_sim_infer_msg_v1_RCCommand to_idl(const RCCommand& value);
RCCommand from_idl(const robot_learning_lab_sim_infer_msg_v1_RCCommand& value);
robot_learning_lab_sim_infer_msg_v1_MotorCommand to_idl(const MotorCommand& value);
MotorCommand from_idl(const robot_learning_lab_sim_infer_msg_v1_MotorCommand& value);

}  // namespace rll_policy::detail
