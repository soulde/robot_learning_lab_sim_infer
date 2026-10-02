# Optional Second Policy Sim2Sim Design

## Goal

Extend `robot_learning_lab_sim_infer` so a profile can run a two-policy handoff/state machine while preserving the existing single-policy runner as the default behavior and a strict subset of the new capability.

## Confirmed requirements

- The selected profile supplies all policy configuration, including checkpoint paths and per-policy observation/action dimensions; there are no checkpoint CLI arguments.
- A profile declares a required primary policy and may declare a secondary policy. With only a primary policy, the existing single-policy behavior is retained and policy-switch state-machine callbacks are disabled.
- Declaring a secondary policy enables profile-directed routing between the two actors. The runner must not invent robot-specific transitions or key bindings.
- The velocity/stand actor is the Chocolate TorchScript export selected by the user from the latest export available at AMP tag `chocolate-stairs-promising-20260914`: `policy_model_18600.pt`. It accepts 78 observations and returns 23 actions. The tracking actor and its motion file remain profile/application inputs, not bundled package assets.
- Policy observation dimensions may differ. Each policy's input/output dimensions are declared by the profile and validated independently before simulation starts.
- Each policy may specify its own Kp/Kd scale and other policy-specific control parameters. If the secondary policy omits an override, it inherits the primary policy's value. A one-policy profile uses the existing default control parameters.
- No training is required for this sim2sim change.

## Architecture

Make the profile the single source of truth for model and policy parameters. Add a `PolicyConfig` dataclass containing a checkpoint path, positive integer `obs_dim` and `action_dim`, and optional policy-specific control overrides (including Kp/Kd scales). The profile exposes `policies`, a mapping with required `primary` and optional `secondary` `PolicyConfig`s. Relative checkpoint paths resolve relative to the profile module. `run(profile, ...)` loads the declared policies; the CLI has no checkpoint arguments and retains only generic runtime controls such as profile selection, headless mode, step limit, and render rate. The profile's current default control parameters remain the fallback when a policy has no override; secondary policy values fall back to primary overrides and then profile defaults.

When a secondary policy is declared, require an explicit multi-policy profile extension with these members:

- `policies`: mapping containing `primary` and `secondary` `PolicyConfig` values;
- `active_policy(model, data, state) -> str | None`: returns `primary`, `secondary`, or `None` once per policy tick;
- `observe_for_policy(policy_name, model, data, state) -> np.ndarray`;
- `apply_policy_action(policy_name, model, data, action, state) -> None`, called each physics substep when an actor is active;
- `apply_transition_control(model, data, state) -> None`, called each physics substep when no actor is active;
- optional `handle_policy_key(key, model, data, state)`, called only when a secondary policy is configured and used for multi-policy transitions.

The current `handle_key(key, model, data, state)` callback remains routed as today in both single- and two-policy modes. The profile owns multi-policy transitions in `handle_policy_key`; the runner does not invent robot-specific transitions or key bindings. That callback is available only when a secondary policy is declared, so a single-policy profile cannot switch policies. The profile may select no actor during a transition/ready phase and provide a profile-owned no-policy control step (for example, moving to and holding the motion's first frame). This allows a consumer profile to implement `VELOCITY_STAND -> TRACKING_READY -> TRACKING_PLAYING` without embedding Chocolate semantics in the generic runner.

The runner owns policy loading and validation, dispatches the selected actor, and continues to own MuJoCo stepping, rendering, timing, and viewer lifecycle. It does not inspect checkpoint filenames or infer model dimensions. Profiles remain dynamically loaded Python files as today.

## API and validation behavior

- CLI: no checkpoint arguments; policy checkpoints and dimensions are loaded only from the selected profile.
- Python: `run(profile, *, headless=False, steps=0, render_hz=25.0)`; no policy checkpoint arguments.
- Single policy: profile declares only `policies.primary`; retain current observation/action behavior and default control parameters. Multi-policy callbacks are not required or called.
- Two policies: fail before building the MuJoCo scene if the profile lacks the multi-policy contract, either actor's TorchScript load fails, its declared shape is wrong, or a dummy forward returns non-finite values.
- Validate each actor with its own declared `(obs_dim, action_dim)`; reject unknown active policy names and invalid/non-finite runtime observations/actions with step and policy context.
- Existing key events continue to reach `handle_key` whether or not a secondary policy is present. `handle_policy_key` is called only when `policies.secondary` exists, so a primary-only profile cannot enter policy-switch states. Reset remains supported and resets profile state and policy action history.
- Explicit policy control overrides take precedence. Secondary values fall back to primary overrides, then to the profile's existing defaults. The resolved values are made available to profile action logic; generic runner code does not assume a fixed set of robot-specific control fields.

## State ownership and Chocolate handoff

The generic inference package exposes policy routing and optional transition-control hooks only. The Chocolate profile owns motion loading, the velocity-zero-command stand state, alignment to motion frame zero, ready/hold behavior, the play key, motion-time advancement, and end-of-clip behavior. The selected AMP export is configured as the primary checkpoint in the consumer's `sim2sim_profile`; no weight is copied into the inference package. The tracking checkpoint, motion file, and any distinct Kp/Kd scale overrides are likewise profile configuration.

The initial Chocolate key mapping will preserve the existing `R` reset behavior, use `T` to align to the first motion frame, and `P` to begin playback from frame zero. Alignment uses a configurable 1.0 second default; while ready, the profile holds frame zero until `P`. Playback advances at the motion's native timestep, does not loop, and holds the final frame until `R` resets to velocity stand. These defaults must not affect single-policy users.

## Error handling

- A configured secondary policy without the required multi-policy profile callbacks: clear startup error naming the missing contract.
- Missing/invalid checkpoint, mismatched dimensions, non-finite model output, invalid policy key, or invalid runtime observation: fail with checkpoint/policy/step context before unsafe action application where possible.
- Without a secondary policy, do not invoke multi-policy callbacks; keep existing single-policy key handling and output behavior unchanged.

## Testing

- Regression test: a profile with one `primary` policy config runs the single-policy path without multi-policy members and retains default controls.
- Profile-config tests: missing/invalid checkpoint config errors clearly; relative checkpoint paths resolve from the profile module; no checkpoint is accepted from the CLI.
- Optional-secondary tests: omitting `secondary` leaves routing disabled; declaring it requires the multi-policy extension. Verify per-policy control overrides, and that secondary Kp/Kd scales and other omitted control values inherit primary values.
- Validate two TorchScript actors with different observation dimensions and the same or different action dimensions.
- Validate malformed TorchScript, shape mismatch, non-finite dummy output, invalid policy selection, and non-finite runtime observation handling.
- Test profile-owned transition/ready/play dispatch and that transition steps do not advance tracking time before the play state.
- Run a headless finite-step test with fixture policies; no RL training is part of validation.

## Non-goals

- No Chocolate model, dataset, keyboard mapping, reward, or training code inside the generic inference package.
- No edits to the AMP repository or the parent `robot_lab` Docker/DR02 working changes.
- No new inference backend, critic loading, or policy architecture inference from ordinary training checkpoints.
- No requirement to train either policy.
