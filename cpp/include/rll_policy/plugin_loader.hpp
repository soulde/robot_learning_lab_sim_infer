#pragma once

#include "rll_policy/processors.hpp"

#include <filesystem>
#include <memory>

namespace rll_policy {

class RobotPluginHandle {
 public:
  static RobotPluginHandle load(const std::filesystem::path& plugin_path,
                                const std::filesystem::path& config_path,
                                MotorCommandWriteFn writer, void* writer_context);
  RobotPluginHandle(RobotPluginHandle&& other) noexcept;
  RobotPluginHandle& operator=(RobotPluginHandle&& other) noexcept;
  RobotPluginHandle(const RobotPluginHandle&) = delete;
  RobotPluginHandle& operator=(const RobotPluginHandle&) = delete;
  ~RobotPluginHandle();

  InputProcessor& input() const { return *input_; }
  OutputProcessor& output() const { return *output_; }

 private:
  RobotPluginHandle() = default;
  void reset() noexcept;
  void* library_{};
  InputProcessor* input_{};
  OutputProcessor* output_{};
  void (*destroy_input_)(InputProcessor*){};
  void (*destroy_output_)(OutputProcessor*){};
};

}  // namespace rll_policy
