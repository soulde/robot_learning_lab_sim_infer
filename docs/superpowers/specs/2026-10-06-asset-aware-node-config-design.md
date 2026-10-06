# Asset-Aware Node Configuration Design

## Goal and agreed boundaries

This package has two separately configured processes: the MuJoCo + Viser simulated hardware node and the native C++ policy node. Each node owns its runtime settings and reads its own YAML file. Robot identity and shared robot defaults come from the robot asset package's single `description/profile.json`; the sim node loads the robot MJCF named by its own config. No ROS or Boost dependency is introduced.

The MJCF actuator force range is the physical torque saturation used by MuJoCo. The sim node must not add a second, RC-mode-specific software effort envelope. Policy YAML may override Kp/Kd for a policy, but it cannot override the MJCF physical force limit. If both the profile and MJCF expose an effort value, the profile value is descriptive/shared asset metadata and must agree with the MJCF value; MuJoCo's actuator model remains the simulation clamp.

Profile Kp/Kd values are defaults. A policy entry may override either gain by joint name. Missing policy overrides inherit the asset profile value. Other policy-specific model facts, such as checkpoint, observation/action dimensions, and optional tracking motion file, stay in policy YAML.

## Sim hardware YAML

The sim config owns simulator and hardware-emulator runtime behavior. Its fields are:

| Field | Meaning |
|---|---|
| `robot.mjcf` | Required path to the authoritative robot MJCF. Resolve relative paths from the YAML file. |
| `robot.profile` | Required path to that asset's `description/profile.json`; used for canonical name/order and consistency checks, not as an alternate torque clamp. |
| `simulation.timestep_s` | Optional MuJoCo step override. Null follows the MJCF timestep. |
| `simulation.sensor_publish_hz` | Robot state and RC publish rate; default 100 Hz. |
| `dds.domain_id`, `dds.cyclonedds_uri` | Cyclone DDS domain and optional interface/config URI. |
| `dds.state_topic`, `dds.rc_topic`, `dds.motor_command_topic` | Shared topic contract; values must match the policy YAML. |
| `dds.sensor_reliability`, `dds.command_reliability` | QoS settings; default best effort for state/RC and reliable for motor commands. |
| `viser.enabled`, `viser.host`, `viser.port` | Viser server settings. |
| `viser.render_hz`, `viser.camera_distance` | Render update rate and initial camera framing. |
| `rc.key_axes`, `rc.key_steps`, `rc.axis_ranges` | Keyboard increments and clamped RC command ranges. |
| `rc.clear_key`, `rc.mode_keys`, `rc.button_keys` | Clear, mode, and button mappings published as RC data. |

The sim config does not contain robot joint order, gains, or effort limits. Joint and body names are checked against the asset profile and MuJoCo model at startup. The renderer reads the in-process `MjModel`/`MjData`; no rendered state is sent over DDS.

## Native policy YAML

One policy YAML owns the policy process settings and robot plugin settings. The plugin-specific values currently kept in a second `chocolate_plugin.yaml` are nested under `plugin.config` so launching the policy process requires one policy YAML.

| Field | Meaning |
|---|---|
| `dds.*` | Domain, optional Cyclone URI, three topic names, and compatible QoS settings. |
| `timing.policy_hz` | Inference/control publication rate; the current Chocolate runtime uses 50 Hz. |
| `timing.state_timeout_s`, `timing.rc_timeout_s` | Maximum age of required input samples. |
| `timing.transition_duration_s` | Output-space blend time when switching policy/state. |
| `device` | LibTorch device, initially `cpu` or a configured CUDA device. |
| `plugin.library` | Shared-library path for the robot's input/output processor. |
| `plugin.asset_profile` | Path to the robot's unified profile JSON. |
| `damping.kd` | Fixed damping-state Kd; damping publishes zero target and Kp. |
| `policies[]` | Entries containing `index`, `name`, `type`, `checkpoint`, `observation_dim`, and `action_dim`; tracking entries may also define `motion_file`, `expected_fps`, and `motion_start_button_mask`. |
| `policies[].gains.kp`, `policies[].gains.kd` | Optional per-joint maps. Resolve each gain as policy override, then profile default. |
| `state_machine` | Initial state, state-to-policy mapping, transitions, RC mode events, and button events. |

Policy YAML has no `effort_limit` field. A plugin must reject unknown policy joints and invalid gain values. Action scaling defaults to the asset profile's `control.action_scale_rad`; an explicit per-policy action-scale override may be supplied only when the exported policy requires a different scale. State transition blending applies to the final command values, including Kp/Kd, and does not change the MJCF actuator limit.

## Illustrative YAML shape

The examples show field ownership, not a checked-in deployment config. Paths to MJCF, profile, checkpoints, and motion data remain deployment inputs.

```yaml
# sim_hardware.yaml
robot:
  mjcf: /path/to/Chocolate.xml
  profile: /path/to/description/profile.json
simulation:
  timestep_s: null
  sensor_publish_hz: 100
dds:
  domain_id: 23
  cyclonedds_uri: null
  state_topic: robot/state
  rc_topic: robot/rc_command
  motor_command_topic: robot/motor_command
  sensor_reliability: best_effort
  command_reliability: reliable
viser:
  enabled: true
  host: 0.0.0.0
  port: 8080
  render_hz: 30
  camera_distance: 2.5
rc:
  key_axes: {I: [vx, 1], K: [vx, -1], J: [vy, 1], L: [vy, -1]}
  key_steps: {vx: 0.1, vy: 0.1, yaw_rate: 0.1}
  axis_ranges: {vx: [-4, 4], vy: [-4, 4], yaw_rate: [-1.5, 1.5]}
  clear_key: Z
  mode_keys: {"0": 0, "1": 1, "2": 2, "3": 3}
  button_keys: {T: 1}
```

```yaml
# policy_node.yaml
dds:
  domain_id: 23
  cyclonedds_uri: null
  state_topic: robot/state
  rc_topic: robot/rc_command
  motor_command_topic: robot/motor_command
  sensor_reliability: best_effort
  command_reliability: reliable
timing:
  policy_hz: 50
  state_timeout_s: 0.25
  rc_timeout_s: 0.5
  transition_duration_s: 0.5
device: cpu
plugin:
  library: /path/to/libchocolate_robot_plugin.so
  asset_profile: /path/to/description/profile.json
damping:
  kd: 0.5
policies:
  - index: 1
    name: velocity
    type: velocity
    checkpoint: /path/to/velocity.pt
    observation_dim: 78
    action_dim: 23
  - index: 2
    name: tracking
    type: tracking
    checkpoint: /path/to/tracking.pt
    observation_dim: 124
    action_dim: 23
    motion_file: /path/to/motion.rllchoc
    expected_fps: 50
    motion_start_button_mask: 1
    gains:
      kp: {left_hip_pitch_joint: 40.0}
state_machine:
  initial_state: damping
  mode_events:
    0: select_damping
    1: select_velocity
    2: select_tracking
  states:
    damping:
      policy: null
      transitions: {select_damping: damping, select_velocity: velocity, select_tracking: tracking}
    velocity:
      policy: velocity
      transitions: {select_damping: damping, select_velocity: velocity, select_tracking: tracking}
    tracking:
      policy: tracking
      transitions: {select_damping: damping, select_velocity: velocity, select_tracking: tracking}
```

## Asset and current-source integration notes

The Chocolate checkout now exposes one `description/profile.json` with 23 controlled joints, 28 bodies, canonical and tracking orders, joint defaults, control gains, action scale, armature, and one control effort-limit map. The sim and policy consumers should use its names instead of carrying separate copies of `joints.json`, `bodies.json`, default joint positions, or default Kp/Kd.

The current Chocolate MJCF has 23 direct `<motor>` actuators but does not yet set `forcelimited` or `forcerange`. Therefore, at the time this design was inspected, the requested source MJCF torque clamp is not active in that XML. The asset MJCF must declare each actuator's force range before the sim node can rely on MuJoCo for torque saturation. The sim adapter will then leave actuator saturation to MuJoCo and remove its mode-indexed `rll_effort_limit_mode_*` lookup and software clipping.

The skill's example schema is not currently copied to the Chocolate asset's `description/schemas/` directory, while the current `profile.json` includes fields beyond that example schema (including `joint_groups`, `joint_orders`, per-joint defaults, and tracking metadata). Asset schema validation and field compatibility must be reconciled as part of asset-package migration; node configs must not invent another profile format.

## Startup validation and failure behavior

- Both nodes require the same DDS domain and topic/type contract; a mismatch is reported before the control loop starts.
- The sim node fails startup if the MJCF/profile pair has missing or duplicate actuated joints, profile references do not resolve, a direct motor is absent, or an actuator has no finite symmetric force limit.
- The policy node fails startup if the profile cannot be loaded, a checkpoint or plugin is missing, a policy dimension is invalid, or gain/action maps reference unknown joints.
- Policy Kp/Kd resolution is performed once at policy load: profile defaults are copied, then only explicitly supplied policy gain keys are overlaid.
- Invalid/stale sensor input withholds active motor output; no effort cap is silently invented in the policy node.

## Acceptance

1. Exactly two runtime YAMLs are sufficient to launch the two nodes; robot files and policy checkpoints are external paths/assets, not extra node config files.
2. The asset profile is the only source of canonical names, joint/body ordering, default joint pose, and default gains.
3. MuJoCo actuator `forcerange` is the only simulated physical torque saturation and is not selected by RC mode or policy index.
4. A per-policy Kp/Kd override changes only that policy's command gains; absent entries inherit asset defaults.
5. No policy or sim config duplicates torque-limit values.
