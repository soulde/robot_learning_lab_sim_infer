#include "rll_policy/config.hpp"
#include "rll_policy/dds_transport.hpp"
#include "rll_policy/inference_engine.hpp"
#include "rll_policy/plugin_loader.hpp"
#include "rll_policy/runtime_utils.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
std::atomic_bool stop_requested{false};
void stop_handler(int) { stop_requested.store(true); }
struct DdsWriterContext { rll_policy::DdsTransport* transport{}; };
void publish_motor(void* raw,const rll_policy::MotorCommand* command) {
  auto* context=static_cast<DdsWriterContext*>(raw);
  if (!context || !context->transport) throw std::runtime_error("DDS writer is not initialized");
  context->transport->publish_motor_command(*command);
}

const rll_policy::PolicySlot& find_slot(const rll_policy::PolicyRuntimeConfig& config,const std::string& name) {
  for (const auto& slot:config.policies) if (slot.model.name==name) return slot;
  throw std::runtime_error("FSM selected unknown policy '"+name+"'");
}
std::optional<std::string> policy_for(const rll_policy::PolicyRuntimeConfig& config,const std::string& state) {
  return config.state_machine.state_policies.at(state);
}
std::uint64_t wall_time_ns() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}
rll_policy::MotorCommand infer_command(const std::string& name,const rll_policy::RobotState& state,
                                       const rll_policy::RCCommand& rc,rll_policy::RobotPluginHandle& plugin,
                                       rll_policy::InferenceEngine& inference,std::uint64_t timestamp) {
  auto observation=plugin.input().build_observation(name,state,rc);
  auto action=inference.infer(name,observation);
  auto command=plugin.output().make_command(name,action,state,timestamp);
  (void)rll_policy::blend_motor_commands(command,command,0.0,timestamp);
  return command;
}
void usage() {
  std::cout << "Usage: rll-policy --config <policy.yaml> [--domain-id <id>] [--cyclonedds-uri <uri>]\n";
}
}

int main(int argc,char** argv) {
  if (argc==2 && std::string(argv[1])=="--help") { usage(); return 0; }
  try {
    std::filesystem::path config_path;
    std::optional<std::uint32_t> domain_override;
    std::optional<std::string> uri_override;
    for (int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if (arg=="--config" && i+1<argc) config_path=argv[++i];
      else if (arg=="--domain-id" && i+1<argc) {
        const auto value=std::stol(argv[++i]); if (value<0) throw std::invalid_argument("domain id must be non-negative");
        domain_override=static_cast<std::uint32_t>(value);
      } else if (arg=="--cyclonedds-uri" && i+1<argc) uri_override=argv[++i];
      else if (arg=="--help") { usage(); return 0; }
      else throw std::invalid_argument("unknown or incomplete argument '"+arg+"'");
    }
    if (config_path.empty()) throw std::invalid_argument("--config is required");
    auto config=rll_policy::PolicyRuntimeConfig::load(config_path);
    if (domain_override) config.dds.domain_id=*domain_override;
    if (uri_override) config.dds.cyclonedds_uri=*uri_override;

    DdsWriterContext writer_context;
    auto plugin=rll_policy::RobotPluginHandle::load(config.plugin_path,config.plugin_config,publish_motor,&writer_context);
    std::vector<rll_policy::PolicyModelConfig> models;
    for (const auto& policy:config.policies) models.push_back(policy.model);
    rll_policy::InferenceEngine inference;
    inference.load_models(models,config.device);
    rll_policy::DdsTransport dds(config.dds);
    writer_context.transport=&dds;
    rll_policy::RuntimeStateMachine state_machine(config.state_machine);

    std::optional<rll_policy::RobotState> robot_state;
    std::optional<rll_policy::RCCommand> rc_command, previous_rc;
    double state_received=-1.0, rc_received=-1.0;
    std::string stable_state=state_machine.state(), transition_target;
    std::optional<std::string> transition_source_policy;
    std::optional<rll_policy::MotorCommand> transition_fixed_source,last_emitted;
    bool transition_active=false;
    Clock::time_point transition_start{};
    const auto period=std::chrono::duration<double>(1.0/config.policy_hz);
    auto next_tick=Clock::now();
    std::signal(SIGINT,stop_handler); std::signal(SIGTERM,stop_handler);
    std::cout << "rll-policy: domain=" << config.dds.domain_id << " policies=" << config.policies.size()
              << " rate_hz=" << config.policy_hz << " device=" << config.device << std::endl;

    while (!stop_requested.load()) {
      const auto loop_start=Clock::now();
      const auto steady_s=std::chrono::duration<double>(loop_start.time_since_epoch()).count();
      if (auto sample=dds.take_latest_state()) { robot_state=std::move(sample); state_received=steady_s; }
      if (auto sample=dds.take_latest_rc()) {
        rc_command=std::move(sample); rc_received=steady_s;
        if (!previous_rc) {
          state_machine.on_mode(rc_command->mode);
          state_machine.on_button(0,rc_command->button_mask);
        } else if (previous_rc->mode!=rc_command->mode) {
          state_machine.on_mode(rc_command->mode);
        } else {
          state_machine.on_button(previous_rc->button_mask,rc_command->button_mask);
        }
        previous_rc=rc_command;
      }

      const bool fresh=rll_policy::active_inputs_fresh(rc_command && rc_command->enabled,
          steady_s,state_received,rc_received,config.state_timeout_s,config.rc_timeout_s,
          robot_state.has_value(),rc_command.has_value());
      if (!fresh) {
        transition_active=false; transition_source_policy.reset(); transition_fixed_source.reset();
        stable_state=state_machine.state(); last_emitted.reset();
      } else {
        const auto desired_state=state_machine.state();
        if (desired_state!=(transition_active?transition_target:stable_state)) {
          std::cout << "[state] " << (transition_active?transition_target:stable_state)
                    << " -> " << desired_state << std::endl;
          const bool interrupted=transition_active;
          transition_target=desired_state; transition_start=loop_start;
          transition_active=config.transition_duration_s>0.0;
          transition_source_policy.reset(); transition_fixed_source.reset();
          if (transition_active) {
            if (interrupted) transition_fixed_source=last_emitted;
            else transition_source_policy=policy_for(config,stable_state);
            if (!transition_source_policy && !transition_fixed_source) {
              transition_fixed_source=last_emitted ? last_emitted : std::optional<rll_policy::MotorCommand>(
                  rll_policy::make_damping_command(*robot_state,config.damping_kd,wall_time_ns()));
            }
          } else stable_state=desired_state;
        }

        const auto now_ns=wall_time_ns();
        rll_policy::MotorCommand output;
        if (transition_active) {
          auto target_policy=policy_for(config,transition_target);
          auto target=target_policy ? infer_command(*target_policy,*robot_state,*rc_command,plugin,inference,now_ns)
                                    : rll_policy::make_damping_command(*robot_state,config.damping_kd,now_ns);
          rll_policy::MotorCommand source;
          if (transition_source_policy) source=infer_command(*transition_source_policy,*robot_state,*rc_command,plugin,inference,now_ns);
          else source=*transition_fixed_source;
          const double raw=std::chrono::duration<double>(loop_start-transition_start).count()/config.transition_duration_s;
          const double p=std::clamp(raw,0.0,1.0); const double alpha=p*p*(3.0-2.0*p);
          output=rll_policy::blend_motor_commands(source,target,alpha,now_ns);
          plugin.output().publish(output); last_emitted=output;
          if (raw>=1.0) { stable_state=transition_target; transition_active=false; transition_source_policy.reset(); transition_fixed_source.reset(); }
        } else {
          const auto policy=policy_for(config,stable_state);
          output=policy ? infer_command(*policy,*robot_state,*rc_command,plugin,inference,now_ns)
                        : rll_policy::make_damping_command(*robot_state,config.damping_kd,now_ns);
          plugin.output().publish(output); last_emitted=output;
        }
      }

      next_tick+=std::chrono::duration_cast<Clock::duration>(period);
      const auto now=Clock::now();
      if (next_tick>now) std::this_thread::sleep_until(next_tick);
      else next_tick=now;
    }
    dds.close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rll-policy fatal: " << error.what() << std::endl;
    return 1;
  }
}
