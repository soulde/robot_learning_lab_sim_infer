# Cyclone DDS 节点化架构设计

## 目标

将当前 sim-to-sim 工具演进为两个独立进程：一个 MuJoCo + Viser 仿真硬件节点和一个策略推理节点。两者只通过本项目定义的 DDS IDL 消息通信，DDS 实现固定为 Eclipse Cyclone DDS，不依赖 ROS 2，也不直接复用宇树或云深处的消息定义。

首阶段以节点边界和可工作的传感器/命令闭环为目标；不实现厂商真机适配，不引入复杂状态机或多机器人支持。

## 架构

```text
Browser
  │ Viser UI (keyboard / mock RC)
  ▼
sim_hardware_node.py (one process, three owned worker threads)
  ├── Simulation thread: sole writer of MjData; applies latest motor command
  ├── Render thread: Viser reads the same in-process MjModel / MjData
  └── DDS thread: publishes robot state + RC and receives motor commands
           │ Cyclone DDS
           ▼
Independent Policy Node
  ├── subscribes robot state and RC command
  ├── builds observations and runs deploy policy
  └── publishes motor command
```

The current package is a small single-process TorchScript/MuJoCo runner with a `Sim2SimProfile` contract; it has no backend/deploy-wrapper abstraction yet. Keep that CLI operational while introducing separate profile contracts for the two nodes. The sim profile owns MuJoCo-specific sensor extraction and motor-command application. The policy profile owns observation construction, policy inference, and the state machine; it must not receive `MjModel` or `MjData`.

## Nodes and responsibilities

### Sim Hardware Node

- The central implementation is `nodes/sim_hardware_node.py`; it starts, supervises, and joins exactly three application worker threads: simulation, rendering, and DDS I/O. Viser/Cyclone may manage their own internal service threads.
- Simulation thread owns stepping and all mutation of `MjData`; it reads a thread-safe latest motor-command slot and applies safe behavior when the command is stale.
- Render thread updates the Viser MuJoCo scene from the same model/data. A shared lock protects reads against simulation writes; renderer updates never step physics.
- DDS I/O thread publishes timestamped robot-state observations and the latest mock RC command, and drains incoming motor commands into the latest-value slot. DDS callbacks never touch MuJoCo.
- Loads an MJCF path from package-owned Hydra structured config and uses the built-in XML hardware profile to index named scalar joints, direct motor actuators, and declared IMU sensors. Package-owned Hydra config supplies DDS, Viser, keyboard/RC ranges, and simulation timing; supported XMLs need no per-robot Python cfg.
- Viser GUI input updates a thread-safe RC input state; it does not interpret runtime states or select/switch policies.
- Generic state-machine and policy-switching config is package-owned, but loaded and executed only by the independent policy node.
- Starts in a safe disabled/zero-command state. On stale commands, applies the configured safe behavior (initially zero torque or controller-defined damping; exact default must be settled during implementation against current backend semantics).

### Policy Node

- Uses package-owned Python runtime config and an externally supplied policy profile for checkpoints, dimensions, observations, and action-to-motor-command conversion; accepts common launch overrides from CLI.
- Subscribes to robot state and RC command; the external policy profile builds observations and runs the configured policy. The existing `Sim2SimProfile` runner remains available for backward compatibility and is not reused across the process boundary because it passes MuJoCo objects to callbacks.
- Owns and runs the configured generic state machine. State transitions, policy selection/switching, and any transition behavior happen here based on robot state and RC input.
- Runs at configured policy rate and publishes a motor command containing explicit control mode and per-joint targets/gains/feedforward effort.
- Uses message joint names for mapping and validates expected joint set/order at startup.
- If required state is absent or stale, does not publish active commands; logs a clear diagnostic.

## First message set

Define project-owned IDL types, versioned under a project namespace. Initial topics:

| Topic | Direction | Content |
|---|---|---|
| `robot/state` | Sim → Policy | timestamp, joint names/position/velocity/effort, base pose and velocity, IMU, contact/foot force |
| `robot/rc_command` | Sim → Policy | timestamp, enable, mode, planar velocity, yaw rate, body height, optional buttons |
| `robot/motor_command` | Policy → Sim | timestamp, joint names, control mode, position/velocity targets, kp/kd, feedforward torque |

Use OMG IDL source files as the message contract and generate Python bindings with Cyclone DDS `idlc -l py`; do not hand-author Python-only DDS dataclasses. Arrays are accompanied by joint names; consumers must not assume that numeric index alone defines a robot joint. Units and coordinate conventions are documented in IDL comments. Use bounded arrays or sequences supported by the compiler.

Initial QoS should favor latest-value robot state and RC data (best effort, keep-last depth 1); motor commands should use reliable delivery, keep-last depth 1, and freshness checked by source timestamp/receipt time. QoS compatibility will be documented and configurable only where needed.

## Configuration style

Use Hydra structured configs declared as Python dataclasses, with CLI dot-path overrides. Keep the MuJoCo model in its normal MJCF XML asset. Configuration has two layers:

- Package-owned Hydra config classes define reusable runtime behavior: state-machine states/transitions and policy switching rules (executed by the policy node), keyboard bindings and physical RC ranges, MuJoCo step and sensor publish periods, DDS topic/QoS/domain defaults, Cyclone DDS XML config URI for network-interface selection, and Viser behavior.
- MJCF defines robot geometry, joint topology, motor actuators, and explicit sensors. The generic sim profile supports one direct motor actuator per named scalar hinge/slide joint and needs no Python hardware cfg. The external policy profile defines checkpoints, dimensions, and robot-specific observation/action adaptations. No MuJoCo objects pass across DDS.

The sim node composes package Hydra defaults and CLI overrides, including the MJCF path. `deploy.json` remains the existing policy observation/action contract loaded by `DeployConfig`; the policy config references it. The policy node continues to accept an external policy profile. Cyclone's network interface is selected through its XML configuration and must be applied before creating the process's first DDS participant.

## Package layout and commands

Proposed package layout:

```text
src/robot_learning_lab_sim_infer/
  dds/                 # IDL definitions, generated-type loading, topic/QoS constants
  nodes/
    sim_hardware_node.py # central MuJoCo + Viser + DDS process, three worker threads
    policy_node.py       # independent inference process
  configs/             # package-owned generic runtime config classes
  config_api.py        # composition/validation contract for external robot cfg
```

Expose separate console commands, for example `rll-sim-hardware` and `rll-policy`. The sim command uses package Hydra defaults and accepts an MJCF path; the policy command accepts an externally supplied Python profile. Keep the current single-process validation CLI during migration. Add Cyclone DDS and Hydra support as optional extras so users who only use the existing offline backend do not need them installed.

## Failure behavior

- DDS startup or incompatible IDL/type: fail with the expected topic/type and Cyclone DDS setup in the diagnostic.
- MuJoCo, Viser, config, or checkpoint initialization failure: terminate that node cleanly and release local resources.
- Policy node disconnect or stale motor command: simulator continues stepping and rendering, but applies the configured safe command.
- Sensor/RC publication loss: policy emits no active command until inputs are valid again.

## Verification

Implementation verification should cover IDL generation/import, two local processes discovering each other, state and RC publication, command reception/application, stale-command handling, and a minimal headless policy cycle. Do not start a real training workload. Full GUI verification is optional when no display is available.

## Open implementation decisions

- Confirm the safe behavior for stale motor commands through the XML actuator mapping; the initial direct-motor profile writes zero actuator effort.
- Preserve serialization compatibility through an explicit IDL module/version namespace; breaking changes require a new version.
