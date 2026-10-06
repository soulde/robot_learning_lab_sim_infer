#pragma once

#include "rll_policy/processors.hpp"

#include <cstdint>

namespace rll_policy {
inline constexpr std::uint32_t kRobotPluginAbiVersion = 1;
}

#if defined(__GNUC__)
#define RLL_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define RLL_PLUGIN_EXPORT
#endif

extern "C" {
RLL_PLUGIN_EXPORT std::uint32_t rll_robot_plugin_abi_version();
RLL_PLUGIN_EXPORT rll_policy::InputProcessor* rll_create_input_processor_v1(const char* config_path);
RLL_PLUGIN_EXPORT void rll_destroy_input_processor_v1(rll_policy::InputProcessor* processor);
RLL_PLUGIN_EXPORT rll_policy::OutputProcessor* rll_create_output_processor_v1(
    const char* config_path, rll_policy::MotorCommandWriteFn writer, void* context);
RLL_PLUGIN_EXPORT void rll_destroy_output_processor_v1(rll_policy::OutputProcessor* processor);
}
