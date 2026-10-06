# C++ Policy Node Design

## Goal

Replace the Python policy-node runtime with a native C++ Cyclone DDS process that loads real TorchScript actors through LibTorch. Keep robot-specific sensor-to-observation and action-to-motor-command behavior replaceable through runtime-loaded C++ plugins, while preserving the existing DDS wire contract and policy state-machine ownership.

## Constraints

- The policy runtime is C++; Python is not part of its inference or DDS execution path.
- Use LibTorch and load TorchScript checkpoints with the C++ JIT API.
- Use Cyclone DDS and the existing `dds/idl/robot_v1.idl`; do not define a second message contract.
- Do not depend on ROS, Boost, or a third-party plugin framework.
- Use a small project-owned Linux plugin loader based on `dlopen`/`dlsym` and a versioned factory ABI.
- Keep robot preprocessing, normalization/history construction, joint ordering, action scaling, and actuator mapping in robot plugins, not in the generic inference engine.
- Preserve the Python simulator node and its topics so the native policy node can communicate with it.
- Configuration remains external and YAML-based; package defaults define only generic runtime behavior.

## Architecture

```text
Cyclone DDS state + RC readers
             │ latest fresh samples
             ▼
     input plugin / InputProcessor
             │ [1, observation_dim] float tensor
             ▼
        InferenceEngine ─── TorchScript models (one per policy slot)
             │ action tensor(s)
             ▼
    output plugin / OutputProcessor
             │ MotorCommand(s), blended through transition interval
             ▼
       Cyclone DDS motor writer
```

The C++ executable owns DDS readers/writer, configuration, policy selection, freshness checks, inference scheduling, transition timing, and plugin lifetime. It contains exactly three primary processing classes:

1. `InputProcessor` is the robot-specific interface that receives a typed `RobotState` and `RCCommand`, builds any required history/normalization, and returns a contiguous finite `float32` tensor shaped `[1, observation_dim]` for the requested policy.
2. `InferenceEngine` is generic. It loads and owns all configured `torch::jit::Module` instances, sets evaluation mode/device, runs inference without gradients, and validates input/output shape, dtype, device, and finiteness.
3. `OutputProcessor` is the robot-specific interface that maps policy actions to named joint targets and gains, constructs the existing `MotorCommand` DDS type, and publishes the final command through a writer callback supplied by the runtime.

The node coordinator is a small executable loop rather than a fourth processing class. It owns the finite-state machine and invokes the three classes in order.

## Plugin contract and loading

- A robot plugin is one shared library selected by the policy YAML. It implements the `InputProcessor` and `OutputProcessor` interfaces declared in a public C++ SDK header.
- The shared object exports a C-linkage ABI-version function and C-linkage factories for the two interfaces. The host opens the path with `dlopen`, resolves symbols with `dlsym`, verifies the ABI version, and destroys plugin objects before unloading the library.
- Plugin implementations can use LibTorch for tensor construction. Host and plugin must use the same C++ standard library ABI, compiler family/version compatibility, C++ standard, and LibTorch build. Runtime diagnostics must report plugin path and expected/actual ABI version.
- The plugin receives robot-specific configuration as a file path or opaque configuration text; the generic runtime does not parse robot-specific fields.
- Linux is the initial supported plugin platform. No plugin discovery scan is required; configuration names the exact library path.

## DDS and message compatibility

- CMake invokes the installed Cyclone `idlc` C backend on the existing IDL and compiles generated type support into the native executable.
- The runtime uses Cyclone DDS C APIs from C++ so the C++ node can share generated message definitions with the existing Python node without requiring a separate Cyclone C++ binding package.
- `RobotState`, `RCCommand`, and `MotorCommand` keep the current topic names, type namespace, fields, bounds, and units. The native process creates readers for state/RC and a writer for motor commands with QoS compatible with the sim node.
- The process drains each reader to the newest sample. It performs no active inference or motor publication when required state/RC samples are missing, stale, or RC is disabled.
- On shutdown, DDS entities, plugin objects, TorchScript modules, and shared-library handles are released in dependency order.

## Configuration and policy behavior

- A package-owned example YAML config, parsed by yaml-cpp, supplies DDS domain/topics, policy period, state/RC timeouts, device (`cpu` or configured CUDA device), plugin path, damping gains, transition duration, and policy slots.
- Each policy slot contains a mode/index, name, TorchScript checkpoint path, observation/action dimensions, and per-joint Kp/Kd defaults consumed by the robot output plugin.
- Mode 0 selects damping: zero position/velocity/torque targets, Kp=0, and configured fixed Kd. Modes 1–9 select only configured policy slots; unconfigured modes leave the current state unchanged.
- Switching between policy slots runs both old and new TorchScript networks during the configured transition window. The input plugin builds policy-specific observations; the output plugin builds each policy's named `MotorCommand`; the runtime interpolates matching position, velocity, torque, Kp, and Kd arrays and publishes the result. Joint names and vector lengths must match before blending.
- Switching into or out of damping follows the same output-space interpolation, with the damping command as the endpoint.
- Models are loaded and validated at startup, before DDS command publication. A missing checkpoint, plugin, unsupported device, wrong observation/action dimension, non-finite tensor, or incompatible joint mapping fails with a clear error instead of falling back to zero-action behavior.

## Build and packaging

- Add a CMake C++17 project and a native executable named `rll-policy`.
- Find LibTorch from the active environment's `torch.utils.cmake_prefix_path` (or explicit `CMAKE_PREFIX_PATH`) and locate Cyclone DDS development headers/libraries plus `idlc`.
- Use yaml-cpp for the runtime file, keeping robot-specific parsing inside the plugin. yaml-cpp is the only new general-purpose runtime dependency; ROS and Boost remain excluded.
- Remove the Python console-script binding for `rll-policy` so the command unambiguously starts the C++ executable. Python sim and dummy-control entry points remain unchanged.
- Document the CMake configure/build/run flow and the exact robot plugin ABI contract.

## Failure behavior

- DDS startup, IDL generation, plugin load/symbol/ABI, YAML schema, model load, and device errors fail before the policy loop starts and identify the failing resource.
- Invalid/stale sensor input or disabled RC withholds active motor output.
- Plugin exceptions, invalid tensors, mismatched joint sets, and non-finite outputs stop command publication and surface a fatal diagnostic; they never silently produce a zero action.
- A policy switch starts a new transition from the currently emitted command, so rapid mode changes do not jump back to an obsolete source policy.

## Verification

- Unit-test the native input/inference/output contracts with a test robot plugin and an actual TorchScript fixture. The fixture is test data only; production code has no dummy/zero-action fallback.
- Build the C++ target against the active Isaac Lab EA LibTorch and the system Cyclone development package.
- Run a short native policy process against the existing Python sim node on an isolated DDS domain; verify state/RC reception, TorchScript inference, motor-message publication, policy switching, damping gains, stale-input withholding, and command freshness.
- Run the existing Python test suite to confirm the simulator and legacy single-process runner remain unchanged.

## Scope boundary

The repository does not currently define a deployable robot's exact observation layout, normalization/history, action scaling, or a production checkpoint. Those belong to the selected robot plugin and external YAML. This change provides the native runtime and plugin contract; it must not invent a robot policy contract or bundle a fake checkpoint.
