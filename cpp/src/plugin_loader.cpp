#include "rll_policy/plugin_loader.hpp"

#include "rll_policy/plugin_api.hpp"

#include <dlfcn.h>

#include <stdexcept>
#include <string>

namespace rll_policy {
namespace {
template <typename Function>
Function symbol(void* library, const char* name, const std::filesystem::path& path) {
  dlerror();
  auto* address = dlsym(library, name);
  if (const char* error = dlerror())
    throw std::runtime_error("robot plugin '" + path.string() + "' is missing symbol " + name + ": " + error);
  return reinterpret_cast<Function>(address);
}
}

RobotPluginHandle RobotPluginHandle::load(const std::filesystem::path& plugin_path,
                                           const std::filesystem::path& config_path,
                                           MotorCommandWriteFn writer, void* writer_context) {
  RobotPluginHandle result;
  result.library_ = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!result.library_)
    throw std::runtime_error("failed to load robot plugin '" + plugin_path.string() + "': " + dlerror());
  try {
    const auto abi = symbol<std::uint32_t (*)()>(result.library_, "rll_robot_plugin_abi_version", plugin_path);
    const auto actual = abi();
    if (actual != kRobotPluginAbiVersion)
      throw std::runtime_error("robot plugin '" + plugin_path.string() + "' ABI mismatch: expected " +
                               std::to_string(kRobotPluginAbiVersion) + ", got " + std::to_string(actual));
    const auto create_input = symbol<InputProcessor* (*)(const char*)>(result.library_, "rll_create_input_processor_v2", plugin_path);
    result.destroy_input_ = symbol<void (*)(InputProcessor*)>(result.library_, "rll_destroy_input_processor_v2", plugin_path);
    const auto create_output = symbol<OutputProcessor* (*)(const char*, MotorCommandWriteFn, void*)>(
        result.library_, "rll_create_output_processor_v2", plugin_path);
    result.destroy_output_ = symbol<void (*)(OutputProcessor*)>(result.library_, "rll_destroy_output_processor_v2", plugin_path);
    result.input_ = create_input(config_path.c_str());
    if (!result.input_) throw std::runtime_error("robot plugin '" + plugin_path.string() + "' returned a null input processor");
    result.output_ = create_output(config_path.c_str(), writer, writer_context);
    if (!result.output_) throw std::runtime_error("robot plugin '" + plugin_path.string() + "' returned a null output processor");
    return result;
  } catch (...) {
    result.reset();
    throw;
  }
}

RobotPluginHandle::RobotPluginHandle(RobotPluginHandle&& other) noexcept
    : library_(other.library_), input_(other.input_), output_(other.output_),
      destroy_input_(other.destroy_input_), destroy_output_(other.destroy_output_) {
  other.library_=nullptr; other.input_=nullptr; other.output_=nullptr;
}
RobotPluginHandle& RobotPluginHandle::operator=(RobotPluginHandle&& other) noexcept {
  if (this != &other) {
    reset(); library_=other.library_; input_=other.input_; output_=other.output_;
    destroy_input_=other.destroy_input_; destroy_output_=other.destroy_output_;
    other.library_=nullptr; other.input_=nullptr; other.output_=nullptr;
  }
  return *this;
}
RobotPluginHandle::~RobotPluginHandle() { reset(); }
void RobotPluginHandle::reset() noexcept {
  if (output_ && destroy_output_) destroy_output_(output_);
  if (input_ && destroy_input_) destroy_input_(input_);
  output_=nullptr; input_=nullptr;
  if (library_) dlclose(library_);
  library_=nullptr;
}
}
