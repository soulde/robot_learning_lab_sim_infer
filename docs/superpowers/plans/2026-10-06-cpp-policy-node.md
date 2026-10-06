# Native C++ Policy Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Python `rll-policy` runtime with a native Cyclone DDS and LibTorch process that loads TorchScript policies and delegates robot-specific observation/action contracts to runtime-loaded C++ plugins.

**Architecture:** A C++17 executable owns DDS, freshness checks, policy-state transitions, multi-network blending, and a versioned `dlopen/dlsym` plugin handle. Its three processing interfaces are `InputProcessor`, `InferenceEngine`, and `OutputProcessor`; the robot plugin implements input tensor construction and action-to-message mapping/publication.

**Tech Stack:** C++17, CMake, Cyclone DDS C API and `idlc`, LibTorch TorchScript, yaml-cpp, POSIX `dlopen`/`dlsym`, CTest.

**Spec:** `docs/superpowers/specs/2026-10-06-cpp-policy-node-design.md`

## Global Constraints

- The policy runtime is C++; Python is not part of its inference or DDS execution path.
- Use LibTorch and load TorchScript checkpoints with the C++ JIT API.
- Use Cyclone DDS and the existing `dds/idl/robot_v1.idl`; do not define a second message contract.
- Do not depend on ROS, Boost, or a third-party plugin framework.
- Use a small project-owned Linux plugin loader based on `dlopen`/`dlsym` and a versioned factory ABI.
- Keep robot preprocessing, normalization/history construction, joint ordering, action scaling, and actuator mapping in robot plugins, not in the generic inference engine.
- Preserve the Python simulator node and its topics so the native policy node can communicate with it.
- Configuration remains external and YAML-based; package defaults define only generic runtime behavior.
- Build the native target as C++17 and validate inference with CPU TorchScript fixtures; CUDA is selected only when explicitly configured and available.
- Use a 0.5 s default output-space transition duration, configurable in YAML.
- Use small assertion-based C++ test executables registered with CTest; do not add GoogleTest or another test framework dependency.

## Review Focus

- A TorchScript model returns a tuple, wrong shape, wrong dtype, NaN, or Inf: reject it during startup/inference; test in Task 3.
- A plugin is missing, lacks a required symbol, or reports the wrong ABI version: fail before starting DDS control output; test in Task 2.
- Robot state joint names are reordered, duplicated, missing, or have mismatched vector lengths: reject mapping/blending instead of pairing by index; test in Tasks 2 and 4.
- RC/state data is missing or stale, or RC is disabled: withhold active output; test in Task 4.
- A second mode switch arrives while a blend is in progress: start from the last emitted command and preserve continuous targets/gains; test in Task 4.

---

### Task 1: Native build scaffold and generated DDS message types

**Files:**
- Create: `cpp/CMakeLists.txt`
- Create: `cpp/include/rll_policy/messages.hpp`
- Create: `cpp/src/messages.cpp`
- Create: `cpp/tests/dds_contract_test.cpp`

**Interfaces:**
- Produces: C++ domain messages `RobotState`, `RCCommand`, and `MotorCommand`, with `std::array` for fixed IDL arrays and bounded `std::vector` for IDL sequences; field names, order, units, and enum values match `dds/idl/robot_v1.idl`.
- Produces: CMake target `rll_policy_messages` and native target `rll-policy`, initially buildable as a help/version-only executable.
- `messages.cpp` converts generated Cyclone C structs to/from C++ domain messages; generated structs remain private to DDS transport.

- [x] **Step 1: Write failing contract tests** in `cpp/tests/dds_contract_test.cpp` using CTest and standard assertions for message enum values, conversion round-trip, and rejection of sequence lengths over the IDL bounds.
- [x] **Step 2: Run the contract test** with CTest and confirm it fails because the C++ target/conversion functions do not exist.
- [x] **Step 3: Add CMake generation** using `find_package(CycloneDDS REQUIRED)`, `find_program(idlc REQUIRED)`, and the existing `robot_v1.idl`; find LibTorch and yaml-cpp, set C++17, and link the Cyclone C target.
- [x] **Step 4: Implement the C++ message types and conversions** in `messages.hpp/.cpp`; compile the initial `rll-policy --help` executable without changing the Python command binding yet.
- [x] **Step 5: Run `cmake --build build-cpp --target rll_policy_messages dds_contract_test` and CTest**; expected: IDL conversion/enum/bounds tests pass.
- [x] **Step 6: Commit** as `build: add native policy node scaffold`.

### Task 2: Three processing interfaces and versioned robot plugin loader

**Files:**
- Create: `cpp/include/rll_policy/processors.hpp`
- Create: `cpp/include/rll_policy/plugin_api.hpp`
- Create: `cpp/include/rll_policy/plugin_loader.hpp`
- Create: `cpp/src/plugin_loader.cpp`
- Create: `cpp/tests/test_robot_plugin.cpp`
- Create: `cpp/tests/plugin_loader_test.cpp`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- `InputProcessor::build_observation(std::string_view policy, const RobotState&, const RCCommand&) -> torch::Tensor`.
- `OutputProcessor::make_command(std::string_view policy, const torch::Tensor& action, const RobotState&, uint64_t timestamp_ns) -> MotorCommand`.
- `OutputProcessor::publish(const MotorCommand&) -> void`; its DDS writer callback is supplied by the host.
- `MotorCommandWriteFn` is `void (*)(void* context, const MotorCommand*)`; the output factory receives this function and context, so the plugin can publish without importing the DDS implementation.
- `RobotPluginHandle::load(const std::filesystem::path& plugin_path, const std::filesystem::path& config_path)` verifies ABI v1 and creates both processors; the handle keeps the library loaded until after both plugin objects are destroyed.
- Plugin exports: `uint32_t rll_robot_plugin_abi_version()`, `InputProcessor* rll_create_input_processor_v1(const char* config_path)`, `void rll_destroy_input_processor_v1(InputProcessor*)`, `OutputProcessor* rll_create_output_processor_v1(const char* config_path, MotorCommandWriteFn, void* context)`, and `void rll_destroy_output_processor_v1(OutputProcessor*)`.

- [x] **Step 1: Write failing loader/processor tests** for a test plugin that builds a two-value tensor from state/RC, maps a two-value action to named motor fields, calls the injected writer, destroys plugin instances before unloading, and for missing-library, missing-symbol, and wrong-ABI cases returns clear errors.
- [x] **Step 2: Build and run `plugin_loader_test`**; confirm compile or load tests fail because the interface and loader are absent.
- [x] **Step 3: Implement the processor interfaces and project-owned `dlopen/dlsym` loader** with ABI version checks and destroy-before-unload lifetime.
- [x] **Step 4: Implement the test plugin** using the exact public SDK header and verify it maps real nonzero inputs/actions (no zero-action fallback).
- [x] **Step 5: Run `ctest --test-dir build-cpp -R 'plugin_loader|processor' --output-on-failure`**; expected: valid plugin passes, missing symbols and ABI mismatch are rejected.
- [ ] **Step 6: Commit** as `feat: add versioned robot policy plugin API`.

### Task 3: LibTorch TorchScript inference engine

**Files:**
- Create: `cpp/include/rll_policy/inference_engine.hpp`
- Create: `cpp/src/inference_engine.cpp`
- Create: `cpp/tests/inference_engine_test.cpp`
- Create: `cpp/tests/fixtures/make_torchscript_fixture.py`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- `PolicyModelConfig` contains `name`, `checkpoint`, `observation_dim`, and `action_dim`.
- `InferenceEngine::load_models(configs, device)` loads each `torch::jit::Module`, calls `eval()`, and validates a `[1, observation_dim]` zero-input probe returns a finite float tensor shaped `[1, action_dim]`.
- `InferenceEngine::infer(policy_name, observation) -> torch::Tensor` runs one no-gradient forward and enforces the same input/output contract.

- [x] **Step 1: Write failing tests** for real TorchScript loading, expected output values, multiple models with different input/output sizes, missing/corrupt checkpoints, tuple output, wrong input/output shape, non-finite input/output, and unavailable configured device.
- [x] **Step 2: Run the focused CTest** and confirm the missing inference implementation causes expected failures.
- [x] **Step 3: Add the fixture generator** that writes small traced TorchScript models with deterministic nonzero outputs for tests only.
- [x] **Step 4: Implement `InferenceEngine`** with `torch::jit::load`, device transfer, evaluation mode, no-grad inference, and explicit errors naming the policy/checkpoint.
- [x] **Step 5: Run `ctest --test-dir build-cpp -R inference_engine --output-on-failure`**; expected: all valid and invalid model contract cases pass.
- [x] **Step 6: Commit** as `feat: run TorchScript policies through LibTorch`.

### Task 4: YAML runtime config, DDS loop, FSM and smooth command transitions

**Files:**
- Create: `cpp/include/rll_policy/config.hpp`
- Create: `cpp/src/config.cpp`
- Create: `cpp/include/rll_policy/dds_transport.hpp`
- Create: `cpp/src/dds_transport.cpp`
- Create: `cpp/src/main.cpp`
- Create: `cpp/config/policy_node.yaml`
- Create: `cpp/tests/policy_runtime_test.cpp`
- Create: `cpp/tests/native_sim_interop_test.py`
- Modify: `cpp/CMakeLists.txt`

**Interfaces:**
- `PolicyRuntimeConfig::load(path)` parses DDS domain/topics/QoS, policy rate and timeouts, device, plugin/config paths, damping Kd, `transition_duration_s` (default `0.5`), policy slots/gains, and FSM `initial_state`, state-to-policy mapping, transitions, mode events, and button events.
- `DdsTransport::take_latest_state()`, `take_latest_rc()`, and `publish_motor_command(const MotorCommand&)` map to the existing typed Cyclone topics.
- The runtime loop applies RC mode/button changes to the configured FSM; mode 0 selects damping, mode 1–9 can select only configured slots, unknown modes do not change state, and stale/disabled input withholds active commands.
- Policy switches infer both endpoints while blending all named `MotorCommand` vectors from the last emitted command over the configured duration.
- `native_sim_interop_test.py` launches a small Python helper that runs the existing `SimHardwareNode` headless with enabled test RC values, plus the native C++ node and a motor-command probe as separate processes.
- `main` owns the procedural loop and lifecycle; no fourth processing class is added.

- [x] **Step 1: Write failing runtime tests** for configured FSM events, mode 0 damping (`kp=0`, configured `kd`), configured/unknown mode selection, stale/disabled input withholding, joint-name reordering by name, duplicate/missing names and vector-length rejection, matching-name output blending, policy-to-damping interpolation, and a new switch during an active transition starting from the last emitted command.
- [x] **Step 2: Run the focused CTest** and confirm the runtime cases fail because config/DDS scheduling/state selection do not exist.
- [x] **Step 3: Implement YAML parsing and config validation**; reject missing fields, invalid rates/timeouts, duplicate policy indices, invalid dimensions/gains, and mode mappings to absent slots.
- [x] **Step 4: Implement the Cyclone DDS C reader/writer transport** using the generated C types and QoS compatible with the Python simulator's latest-value sensor/RC and reliable motor-command endpoints.
- [x] **Step 5: Implement the policy loop and transition blending**; use the plugin to build policy-specific observations and per-policy motor commands, then interpolate position, velocity, torque, Kp, and Kd by joint name.
- [x] **Step 6: Run `ctest --test-dir build-cpp -R policy_runtime --output-on-failure`**; expected: state machine, freshness, damping, policy inference routing, and transition tests pass.
- [x] **Step 7: Add the Python simulator helper and native interop test**; the helper constructs the real `SimHardwareNode`, sets `RCValues(enabled=True, mode=1)`, disables rendering, and runs in its own process; the test probes published motor commands.
- [x] **Step 8: Run the native sim interop test** on a dedicated DDS domain; expected: Python sensor/RC samples reach C++, actual TorchScript inference runs, and a nonzero named motor command returns to the simulator.
- [x] **Step 9: Commit** as `feat: add native DDS policy runtime`.

### Task 5: Replace the Python policy entry point and verify sim interoperability

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `src/robot_learning_lab_sim_infer/node_profiles.py`
- Modify: `src/robot_learning_lab_sim_infer/config_api.py`
- Modify: `.gitignore` only if the CMake build directory needs a project-local ignore rule.
- Remove: `src/robot_learning_lab_sim_infer/nodes/policy_node.py`
- Remove: `tests/test_policy_node.py`
- Create: `cpp/tests/install_contract_test.cmake`

**Interfaces:**
- CMake installs the native executable as `rll-policy`; the Python sim and dummy-control commands remain unchanged.
- The config and plugin ABI guide document how a robot-specific shared library and TorchScript checkpoint are selected.
- The interop test launches the existing Python simulator and a C++ native policy process in separate processes on a test DDS domain with the test plugin and TorchScript fixture.

- [x] **Step 1: Write the packaging contract test** to assert CMake installs a native `rll-policy` binary and that `pyproject.toml` no longer registers a Python console script with the same name.
- [x] **Step 2: Run the packaging contract test** and confirm it fails because the Python project still owns the `rll-policy` binding and CMake has no install target.
- [x] **Step 3: Replace the Python `rll-policy` console binding** with CMake installation of the native target, remove the obsolete Python node/test and policy-only Python profile contract, and update README build/run/plugin instructions.
- [x] **Step 4: Run the packaging contract test**; expected: the installed command resolves to the native C++ executable and Python sim/dummy commands remain.
- [x] **Step 5: Re-run the native interop test** on an isolated Cyclone DDS domain; expected: sensor/RC-to-TorchScript-to-MotorCommand round trip passes across C++ and Python processes.
- [x] **Step 6: Run `ctest --test-dir build-cpp --output-on-failure` and the full Python suite `pytest -q`**; expected: all C++ and Python tests pass.
- [x] **Step 7: Commit** as `feat: replace Python policy node with C++ executable`.

## Execution Notes

- The current host has LibTorch CMake files in the Isaac Lab EA Python environment, but the Cyclone DDS runtime development header/library pair was not found under `/usr`; implementation must locate an installed Cyclone DDS development package before building and must not silently use the unrelated private `pico_dds_bridge` build path.
- The test TorchScript fixture is generated locally from installed PyTorch and is never a production fallback or bundled model.
- Live DDS testing requires network-interface access. Run only after the ordinary CMake/CTest checks pass, and request sandbox escalation if Cyclone DDS cannot enumerate interfaces inside the sandbox.

### Task 6: Integrate the Chocolate velocity and whole-body tracking policies

**Files:**
- Create a Chocolate C++ robot plugin implementing both policies' exact observation and named action mapping.
- Create a small offline converter for the existing tracking `.npz` reference into a versioned runtime binary file; do not check in the motion dataset or checkpoints.
- Create a Chocolate policy YAML example referencing both exported TorchScript checkpoints and the generated motion reference.
- Create plugin contract tests for both observation layouts (78 and 124 values), action history, mode transitions, joint order, and `0.5` action-scale position commands.
- Modify the top-level README with Chocolate build, conversion, and run instructions.

**Source of truth:** `/home/jvwei/chocolate_training/sim2sim_profile.py`, Chocolate joint manifest/control profile, existing simulator DDS state contract, and the Chocolate velocity/tracking TorchScript checkpoints. The sample selects velocity on mode 1 and tracking on mode 2; runtime output blending remains owned by the generic C++ coordinator.

- [ ] **Step 1: Capture Chocolate joint orders, scales, gains, default pose, observation terms, and checkpoint contracts in a failing plugin contract test.**
- [ ] **Step 2: Implement the Chocolate plugin and offline motion converter; verify both models against the C++ inference API.**
- [ ] **Step 3: Add the Chocolate YAML example and user-facing invocation; keep model and motion files external.**
- [ ] **Step 4: Run Chocolate plugin contracts and a headless DDS integration smoke test.**
- [ ] **Step 5: Commit** as `feat: add Chocolate tracking policy example`.
