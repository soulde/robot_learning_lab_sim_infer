# Optional Dual-Policy Sim2Sim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the inference package with profile-configured optional dual-policy routing while retaining primary-only inference behavior.

**Architecture:** `PolicyConfig` describes each actor and its optional control overrides; `Sim2SimProfile.policies` is the sole source of checkpoint paths and dimensions. The runner loads and validates configured actors, routes policy ticks through profile callbacks only when a secondary exists, and leaves robot-specific controls/state transitions to the profile. Secondary control values inherit primary overrides and then existing profile defaults.

**Tech Stack:** Python 3.10+, dataclasses, NumPy, PyTorch TorchScript, MuJoCo, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-optional-second-policy-sim2sim-design.md`

## Global Constraints

- All policy checkpoints and per-policy dimensions come from the selected profile; no checkpoint CLI parameters.
- A primary-only profile retains single-policy behavior and does not enable multi-policy switching.
- A configured secondary policy requires profile-owned routing, observation, action, and transition-control callbacks.
- Secondary per-policy control overrides fall back to primary overrides, then to the profile's existing defaults.
- The inference package does not bundle the Chocolate model, motion dataset, or robot-specific state machine.
- No training is required.

## Review Focus

- Profile import/configuration errors, including missing primary policy and malformed configs, should fail with actionable context. Pin in Task 1 tests.
- Relative checkpoint paths must resolve from the profile module, not the caller's working directory. Pin in Task 1 tests.
- TorchScript shape mismatch and non-finite outputs must fail before simulation starts. Pin in Task 2 tests.
- Unknown policy names, wrong-shaped/non-finite runtime observations, and invalid actions must not be applied to MuJoCo. Pin in Task 3 tests.
- A profile with one actor must never enter multi-policy routing, while a secondary actor with omitted control overrides inherits primary values. Pin in Tasks 1 and 3 tests.

---

### Task 1: Profile policy configuration contract

**Files:**
- Modify: `src/robot_learning_lab_sim_infer/profile.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Produces `PolicyConfig(checkpoint: str | Path, obs_dim: int, action_dim: int, control_overrides: Mapping[str, Any] = {})`.
- Profiles expose `policies: Mapping[str, PolicyConfig]` with required `primary` and optional `secondary` entries.
- `load_profile(path)` validates the policy mapping and resolves each checkpoint path relative to the profile file when it is not absolute.
- Secondary resolved control values use precedence: secondary override, primary override, then existing profile defaults. Keep control values generic; the package must not assume robot-specific field names.

- [ ] **Step 1: Write failing profile contract tests**

Test `test_load_profile_requires_primary_policy`, `test_load_profile_rejects_invalid_policy_dimensions`, `test_load_profile_resolves_relative_checkpoints_from_profile`, `test_secondary_control_overrides_inherit_primary_values`, and `test_single_policy_uses_profile_defaults`. Assert actionable exceptions and exact resolved paths/values.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `pytest tests/test_profile.py -q`
Expected: FAIL because the profile contract and validation are not implemented.

- [ ] **Step 3: Implement `PolicyConfig` and profile validation in `profile.py`**

Validate positive integer dimensions and a mapping with `primary` and at most `secondary`; reject unknown policy keys. Resolve checkpoint paths using the loaded profile module's parent directory. Retain profile attributes for existing default controls, and expose a helper that returns merged controls for either actor.

- [ ] **Step 4: Run profile tests**

Run: `pytest tests/test_profile.py -q`
Expected: PASS.

- [ ] **Step 5: Commit the profile contract**

```bash
git add src/robot_learning_lab_sim_infer/profile.py tests/test_profile.py
git commit -m "feat: add profile-owned policy configuration"
```

### Task 2: Profile-only CLI and policy loading validation

**Files:**
- Modify: `src/robot_learning_lab_sim_infer/main.py`
- Modify: `src/robot_learning_lab_sim_infer/runner.py`
- Test: `tests/test_policy_loading.py`

**Interfaces:**
- Produces `run(profile, *, headless=False, steps=0, render_hz=25.0) -> None`.
- CLI accepts `--profile`, `--headless`, `--steps`, and `--render-hz`; it has no checkpoint options.
- `load_torchscript_policy(path, obs_dim, action_dim)` validates TorchScript loading, output shape `(1, action_dim)`, and finite dummy-forward output.

- [ ] **Step 1: Write failing CLI and actor validation tests**

Test that parser invocation rejects `--checkpoint` and `--second-checkpoint`; test loading valid fixture TorchScript actors with different input dimensions; test malformed TorchScript, wrong action shape, and non-finite dummy output errors naming the policy/checkpoint.

- [ ] **Step 2: Run focused tests and confirm they fail**

Run: `pytest tests/test_policy_loading.py -q`
Expected: FAIL because CLI and runner still require one external checkpoint argument.

- [ ] **Step 3: Update the entry point and runner signature**

Change `main()` to print the profile's configured policies and call `run(profile, ...)`. Remove all checkpoint CLI arguments. Update `run` to load primary and optional secondary from `profile.policies` and validate both before calling `profile.build_model()`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_policy_loading.py -q`
Expected: PASS.

- [ ] **Step 5: Commit profile-only model loading**

```bash
git add src/robot_learning_lab_sim_infer/main.py src/robot_learning_lab_sim_infer/runner.py tests/test_policy_loading.py
git commit -m "feat: load sim2sim actors from profile"
```

### Task 3: Optional routing and one-policy behavior

**Files:**
- Modify: `src/robot_learning_lab_sim_infer/profile.py`
- Modify: `src/robot_learning_lab_sim_infer/runner.py`
- Test: `tests/test_runner_routing.py`

**Interfaces:**
- Multi-policy profiles implement `active_policy(model, data, state) -> str | None`, `observe_for_policy(policy_name, model, data, state) -> np.ndarray`, `apply_policy_action(policy_name, model, data, action, state) -> None`, and `apply_transition_control(model, data, state) -> None`.
- `handle_policy_key(key, model, data, state)` is optional and invoked only when `secondary` is configured. Existing `handle_key` behavior remains available in primary-only and dual-policy runs.
- `RuntimeState.action` is sized for the active action dimension and contains only validated finite actions before profile action application.

- [ ] **Step 1: Write failing routing tests**

Test primary-only regression using the legacy `observe`/`apply_action` callbacks; test dual-policy primary-to-secondary selection, distinct observation/action dimensions, `None` transition-control ticks, and secondary-key callback gating. Also assert unknown actor names, invalid observations, and non-finite actions raise before action application.

- [ ] **Step 2: Run routing tests and confirm they fail**

Run: `pytest tests/test_runner_routing.py -q`
Expected: FAIL because runner currently supports one actor only.

- [ ] **Step 3: Implement policy dispatch in `runner.py`**

Keep the primary-only loop on the existing profile callbacks. When a secondary exists, validate the full multi-policy callback contract before model construction, choose one actor per policy tick, call its observation/action callbacks, and call `apply_transition_control` for `None`. Route key events to `handle_policy_key` only in dual-policy mode. Include policy name and step in runtime validation errors.

- [ ] **Step 4: Run routing tests and full package tests**

Run: `pytest tests/test_runner_routing.py -q`
Expected: PASS.

Run: `pytest -q`
Expected: all tests PASS.

- [ ] **Step 5: Commit optional routing**

```bash
git add src/robot_learning_lab_sim_infer/profile.py src/robot_learning_lab_sim_infer/runner.py tests/test_runner_routing.py
git commit -m "feat: add optional dual-policy routing"
```

### Task 4: Consumer integration guidance and headless smoke validation

**Files:**
- Modify: `README.md`
- Test: `tests/test_runner_smoke.py`

**Interfaces:**
- Document profile examples for one actor and two actors, including per-policy dimensions, independent control overrides, secondary inheritance, and the no-checkpoint-CLI invocation.
- A headless finite-step smoke test builds a minimal MuJoCo profile and fixture TorchScript actors, then exercises both primary-only and dual-policy runs.

- [ ] **Step 1: Write failing finite-step smoke tests**

Test `run(profile, headless=True, steps=2)` with one actor and with two actors; assert finite simulation completion and expected policy/transition callback counts.

- [ ] **Step 2: Run smoke tests and confirm they fail or expose integration defects**

Run: `pytest tests/test_runner_smoke.py -q`
Expected: FAIL until the completed runner contract integrates with MuJoCo and fixture profiles.

- [ ] **Step 3: Document profile configuration and invocation**

Update `README.md` with a concise `PolicyConfig` example, primary-only and dual-policy mappings, secondary Kp/Kd-scale inheritance, and a CLI example that passes only `--profile` and runtime controls.

- [ ] **Step 4: Run all tests and static checks**

Run: `pytest -q`
Expected: all tests PASS.

Run: `python -m compileall -q src tests`
Expected: exit code 0.

Run: `git diff --check`
Expected: no whitespace errors.

- [ ] **Step 5: Commit docs and smoke coverage**

```bash
git add README.md tests/test_runner_smoke.py
git commit -m "test: cover profile-driven sim2sim smoke runs"
```
