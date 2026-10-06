#include "rll_policy/config.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <utility>

namespace rll_policy {
namespace {
template <typename T> T required(const YAML::Node& node, const char* name) {
  if (!node[name]) throw std::invalid_argument(std::string("missing config field '") + name + "'");
  try { return node[name].as<T>(); }
  catch (const YAML::Exception& e) { throw std::invalid_argument(std::string("invalid config field '") + name + "': " + e.what()); }
}
std::filesystem::path resolve_path(const std::filesystem::path& base, const std::string& value) {
  std::string expanded=value;
  if (expanded.rfind("~/",0)==0) {
    const char* home=std::getenv("HOME");
    if (!home) throw std::invalid_argument("cannot expand '~': HOME is not set");
    expanded=std::string(home)+expanded.substr(1);
  }
  auto path=std::filesystem::path(expanded);
  return (path.is_absolute() ? path : base/path).lexically_normal();
}
Reliability parse_reliability(const std::string& value, const char* name) {
  if (value == "best_effort") return Reliability::best_effort;
  if (value == "reliable") return Reliability::reliable;
  throw std::invalid_argument(std::string("invalid ") + name + " reliability: " + value);
}
}

PolicyRuntimeConfig PolicyRuntimeConfig::load(const std::filesystem::path& config_path) {
  YAML::Node root;
  try { root=YAML::LoadFile(config_path.string()); }
  catch (const YAML::Exception& e) { throw std::runtime_error("failed to read policy YAML '"+config_path.string()+"': "+e.what()); }
  const auto base=std::filesystem::absolute(config_path).parent_path();
  PolicyRuntimeConfig out;
  auto dds=root["dds"], timing=root["timing"], plugin=root["plugin"], damping=root["damping"], policies=root["policies"], fsm=root["state_machine"];
  if (!dds || !timing || !plugin || !damping || !policies || !fsm) throw std::invalid_argument("policy config must define dds, timing, plugin, damping, policies, state_machine");
  auto domain=required<int>(dds,"domain_id"); if (domain<0) throw std::invalid_argument("dds.domain_id must be non-negative"); out.dds.domain_id=static_cast<std::uint32_t>(domain);
  if (dds["cyclonedds_uri"] && !dds["cyclonedds_uri"].IsNull()) out.dds.cyclonedds_uri=dds["cyclonedds_uri"].as<std::string>();
  out.dds.state_topic=required<std::string>(dds,"state_topic"); out.dds.rc_topic=required<std::string>(dds,"rc_topic");
  out.dds.motor_command_topic=required<std::string>(dds,"motor_command_topic");
  if (out.dds.state_topic.empty() || out.dds.rc_topic.empty() || out.dds.motor_command_topic.empty()) throw std::invalid_argument("DDS topic names cannot be empty");
  out.dds.sensor_reliability=parse_reliability(required<std::string>(dds,"sensor_reliability"),"sensor");
  out.dds.command_reliability=parse_reliability(required<std::string>(dds,"command_reliability"),"command");
  out.policy_hz=required<double>(timing,"policy_hz"); out.state_timeout_s=required<double>(timing,"state_timeout_s"); out.rc_timeout_s=required<double>(timing,"rc_timeout_s");
  if (timing["transition_duration_s"]) out.transition_duration_s=timing["transition_duration_s"].as<double>();
  if (!(std::isfinite(out.policy_hz) && out.policy_hz>0 && std::isfinite(out.state_timeout_s) && out.state_timeout_s>0 &&
        std::isfinite(out.rc_timeout_s) && out.rc_timeout_s>0 && std::isfinite(out.transition_duration_s) && out.transition_duration_s>=0))
    throw std::invalid_argument("policy rates/timeouts must be positive and transition duration non-negative");
  out.device=root["device"] ? root["device"].as<std::string>() : "cpu";
  if (out.device.empty()) throw std::invalid_argument("device cannot be empty");
  out.plugin_path=resolve_path(base,required<std::string>(plugin,"path"));
  out.plugin_config=resolve_path(base,required<std::string>(plugin,"config"));
  out.damping_kd=required<double>(damping,"kd");
  if (!std::isfinite(out.damping_kd) || out.damping_kd<0) throw std::invalid_argument("damping.kd must be finite and non-negative");

  if (!policies.IsSequence() || policies.size()==0) throw std::invalid_argument("policies must be a non-empty sequence");
  std::set<int> indices; std::set<std::string> names;
  for (const auto& item: policies) {
    PolicySlot slot; int index=required<int>(item,"index");
    if (index<1 || index>9 || !indices.insert(index).second) throw std::invalid_argument("policy indices must be unique values from 1 to 9");
    slot.index=static_cast<std::uint16_t>(index); slot.model.name=required<std::string>(item,"name");
    if (slot.model.name.empty() || !names.insert(slot.model.name).second) throw std::invalid_argument("policy names must be non-empty and unique");
    slot.model.checkpoint=resolve_path(base,required<std::string>(item,"checkpoint"));
    slot.model.observation_dim=required<std::int64_t>(item,"observation_dim"); slot.model.action_dim=required<std::int64_t>(item,"action_dim");
    if (slot.model.observation_dim<=0 || slot.model.action_dim<=0) throw std::invalid_argument("policy dimensions must be positive");
    out.policies.push_back(std::move(slot));
  }

  out.state_machine.initial_state=required<std::string>(fsm,"initial_state");
  auto states=fsm["states"]; if (!states || !states.IsMap()) throw std::invalid_argument("state_machine.states must be a mapping");
  for (const auto& item: states) {
    auto state=item.first.as<std::string>(); auto node=item.second;
    std::optional<std::string> policy;
    if (node["policy"] && !node["policy"].IsNull()) policy=node["policy"].as<std::string>();
    if (policy && !names.count(*policy)) throw std::invalid_argument("state '"+state+"' maps to an unknown policy '"+*policy+"'");
    out.state_machine.state_policies.emplace(state,policy);
    auto transitions=node["transitions"]; if (transitions && transitions.IsMap()) {
      auto& dest=out.state_machine.transitions[state];
      for (const auto& edge: transitions) {
        auto event=edge.first.as<std::string>(); auto target=edge.second.as<std::string>();
        if (!dest.emplace(event,target).second) throw std::invalid_argument("duplicate state event '"+event+"'");
      }
    }
  }
  if (!out.state_machine.state_policies.count(out.state_machine.initial_state)) throw std::invalid_argument("state_machine.initial_state is not defined");
  auto modes=fsm["mode_events"]; if (!modes || !modes.IsMap()) throw std::invalid_argument("state_machine.mode_events must be a mapping");
  for (const auto& item: modes) {
    auto mode=item.first.as<int>(); auto event=item.second.as<std::string>();
    if (mode<0 || mode>9 || !out.state_machine.mode_events.emplace(static_cast<std::uint16_t>(mode),event).second) throw std::invalid_argument("mode events must use unique modes 0 through 9");
  }
  auto buttons=fsm["button_events"]; if (buttons) {
    if (!buttons.IsMap()) throw std::invalid_argument("state_machine.button_events must be a mapping");
    for (const auto& item: buttons) {
      auto bit=item.first.as<std::uint32_t>(); auto event=item.second.as<std::string>();
      if (!bit || (bit & (bit-1)) || !out.state_machine.button_events.emplace(bit,event).second) throw std::invalid_argument("button events must use unique single-bit masks");
    }
  }
  for (const auto& [state,edges]: out.state_machine.transitions) {
    if (!out.state_machine.state_policies.count(state)) throw std::invalid_argument("transition source state is undefined: "+state);
    for (const auto& [event,target]: edges) {
      (void)event;
      if (!out.state_machine.state_policies.count(target)) throw std::invalid_argument("transition target state is undefined: "+target);
    }
  }
  if (!out.state_machine.mode_events.count(0)) throw std::invalid_argument("mode 0 must be configured for damping");
  for (const auto& [mode,event]:out.state_machine.mode_events) {
    (void)event;
    if (mode!=0 && !indices.count(mode)) throw std::invalid_argument("mode events may only select configured policy indices");
  }
  const auto damping_event=out.state_machine.mode_events.at(0);
  for (const auto& [state,policy]:out.state_machine.state_policies) {
    auto edges=out.state_machine.transitions.find(state);
    if (edges==out.state_machine.transitions.end() || !edges->second.count(damping_event) ||
        out.state_machine.state_policies.at(edges->second.at(damping_event)).has_value())
      throw std::invalid_argument("mode 0 must transition every state to a damping state");
    (void)policy;
  }
  for (const auto& slot: out.policies) {
    bool mapped=false;
    for (const auto& [state,policy]: out.state_machine.state_policies) if (policy && *policy==slot.model.name) mapped=true;
    if (!mapped) throw std::invalid_argument("policy '"+slot.model.name+"' is not mapped to any state");
    auto event=out.state_machine.mode_events.find(slot.index);
    if (event==out.state_machine.mode_events.end()) throw std::invalid_argument("policy index "+std::to_string(slot.index)+" has no mode event");
    for (const auto& [state,policy]:out.state_machine.state_policies) {
      auto edges=out.state_machine.transitions.find(state);
      if (edges==out.state_machine.transitions.end() || !edges->second.count(event->second) ||
          out.state_machine.state_policies.at(edges->second.at(event->second))!=std::optional<std::string>(slot.model.name))
        throw std::invalid_argument("policy index "+std::to_string(slot.index)+" must select policy '"+slot.model.name+"' from every state");
      (void)policy;
    }
  }
  return out;
}

RuntimeStateMachine::RuntimeStateMachine(StateMachineConfig config): config_(std::move(config)), state_(config_.initial_state) {
  if (!config_.state_policies.count(state_)) throw std::invalid_argument("initial FSM state is not defined: "+state_);
}
std::optional<std::string> RuntimeStateMachine::policy() const { return config_.state_policies.at(state_); }
bool RuntimeStateMachine::apply(const std::string& event) {
  auto state=config_.transitions.find(state_); if (state==config_.transitions.end()) return false;
  auto edge=state->second.find(event); if (edge==state->second.end() || edge->second==state_) return false;
  state_=edge->second; return true;
}
bool RuntimeStateMachine::on_mode(std::uint16_t mode) {
  auto event=config_.mode_events.find(mode); return event!=config_.mode_events.end() && apply(event->second);
}
bool RuntimeStateMachine::on_button(std::uint32_t previous, std::uint32_t current) {
  auto rising=(~previous)&current;
  for (const auto& [bit,event]:config_.button_events) if (rising&bit) return apply(event);
  return false;
}
}
