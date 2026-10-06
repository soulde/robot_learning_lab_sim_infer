# Cyclone DDS node implementation plan

## Goal and constraints

Refactor around `nodes/sim_hardware_node.py`: one MuJoCo + Viser + DDS process with three application threads for simulation, rendering, and DDS publication/command reception. Add an independent policy process. They communicate through project-owned Cyclone DDS messages. Preserve the existing `Sim2SimProfile`/`runner.run()` CLI during migration. The simulation thread alone mutates MuJoCo state. The policy process alone owns state-machine decisions, policy selection, observation construction, and inference.

Use externally supplied Python profile/config classes for robot-specific model, sensors, actuators, checkpoints, and policy adaptations. Package-owned Python classes hold generic DDS, Viser, RC mapping, inference, and state-machine defaults. Do not add ROS 2 or vendor message dependencies.

## Current baseline

`develop` currently contains only `profile.py`, `runner.py`, and the legacy CLI. There is no backend, DeployWrapper, DeployConfig, or state-machine package to reuse. The legacy profile callbacks take `MjModel`/`MjData`, so they remain local to the old runner and cannot serve as the cross-process policy API.

## Implementation sequence

1. **Wire contract and transport**: add versioned OMG IDL for `RobotState`, `RCCommand`, and `MotorCommand`; generated bindings must come from Cyclone DDS `idlc -l py`. Add optional Cyclone dependency, topic/QoS helpers, and lifecycle APIs. Keep importing the base package independent of Cyclone DDS.
2. **Package runtime config and profiles**: add Hydra structured runtime config, an XML hardware profile for named scalar direct-motor joints and declared IMU sensors, and a policy profile contract. Package defaults point to the copied DR02 torque motor XML. State-machine config and execution are policy-side only. RC mapping translates keyboard state into a wire RC sample and makes no policy decisions.
3. **Central sim hardware node**: implement `nodes/sim_hardware_node.py` with one shared model/data pair and three supervised threads. Simulation owns MuJoCo mutation at `runtime.simulation_dt` and applies the latest command or zero-effort safe behavior; rendering runs at the configured Viser rate under a shared data lock; DDS thread publishes state/RC at the configured sensor rate and copies incoming motor commands into a thread-safe latest-value slot. Never mutate MuJoCo from render or DDS threads.
4. **Policy process and migration docs**: consume fresh state/RC samples, build observations with the external policy profile, run its state machine and TorchScript actors, publish named motor commands, and add separate CLI entry points. Retain the old CLI unchanged.
5. **Verification**: verify IDL compilation and typed serialization, Hydra CLI overrides, DR02 model/sensor extraction, Cyclone topic message flow, profile/config/RC/state-machine/command contracts, and legacy runner behavior. Do not start training.

## Review invariants

- State machine and policy switching execute only in the policy node.
- `nodes/sim_hardware_node.py` is the single sim-hardware process entry and owns three worker threads: render, simulation, and DDS data I/O.
- One lock protects every cross-thread access to `MjData`; only the simulation thread may write it.
- No `MjModel` or `MjData` crosses DDS or appears in the policy-node profile callbacks.
- Joint arrays carry names and are mapped/validated explicitly.
- Stale inputs cannot trigger active inference output; stale motor commands select configured safe actuation.
- Optional dependencies do not break legacy CLI installation/import.
