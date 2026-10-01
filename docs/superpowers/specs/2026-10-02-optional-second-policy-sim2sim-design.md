# Optional Second Policy Sim2Sim Design

## Goal

Extend `robot_learning_lab_sim_infer` so a profile can run a two-policy handoff/state machine while preserving the existing single-policy runner as the default behavior and a strict subset of the new capability.

## Confirmed requirements

- `--checkpoint` continues to select the required primary TorchScript actor.
- `--second-checkpoint` is optional. Without it, the current single-policy path runs unchanged and no policy-switch state machine is enabled.
- Supplying a second checkpoint enables profile-directed routing between the two actors. The runner must not invent robot-specific transitions or key bindings.
- The velocity/stand actor is the Chocolate TorchScript export selected by the user from the latest export available at AMP tag `chocolate-stairs-promising-20260914`: `policy_model_18600.pt`. It accepts 78 observations and returns 23 actions. The tracking actor and its motion file remain profile/application inputs, not bundled package assets.
- Policy observation dimensions may differ. Each policy's input/output dimensions are declared by the profile and validated independently before simulation starts.
- Existing one-policy profiles continue to implement only the current `Sim2SimProfile` contract.
- No training is required for this sim2sim change.

## Architecture

Keep `run(profile, checkpoint, ...)` and the current `Sim2SimProfile` interface backward compatible. Add an optional second-checkpoint argument to the Python API and CLI. The one-checkpoint branch keeps the current policy loading, observation, action application, keyboard handling, and timing behavior.

When a second checkpoint is supplied, require an explicit multi-policy profile extension with these members:

- `policy_specs`: mapping containing exactly `primary` and `secondary`, each a `PolicySpec` dataclass with positive integer `obs_dim` and `action_dim` fields;
- `active_policy(model, data, state) -> str | None`: returns `primary`, `secondary`, or `None` once per policy tick;
- `observe_for_policy(policy_name, model, data, state) -> np.ndarray`;
- `apply_policy_action(policy_name, model, data, action, state) -> None`, called each physics substep when an actor is active;
- `apply_transition_control(model, data, state) -> None`, called each physics substep when no actor is active;
- optional `handle_policy_key(key, model, data, state)`, called only when the second checkpoint is enabled and used for multi-policy transitions.

The current `handle_key(key, model, data, state)` callback remains routed exactly as today in both single- and two-policy modes for backward compatibility. The profile owns multi-policy transitions in `handle_policy_key`; the runner does not invent robot-specific transitions or key bindings. The profile may select no actor during a transition/ready phase and provide a profile-owned no-policy control step (for example, moving to and holding the motion's first frame). This allows a consumer profile to implement `VELOCITY_STAND -> TRACKING_READY -> TRACKING_PLAYING` without embedding Chocolate semantics in the generic runner.

The runner owns policy loading and validation, dispatches the selected actor, and continues to own MuJoCo stepping, rendering, timing, and viewer lifecycle. It does not inspect checkpoint filenames or infer model dimensions. Profiles remain dynamically loaded Python files as today.

## API and validation behavior

- CLI: keep required `--checkpoint`; add optional `--second-checkpoint`.
- Python: add keyword-only `second_checkpoint=None` to `run`.
- Single policy: use existing `profile.obs_dim`, `profile.action_dim`, `profile.observe`, and `profile.apply_action`; do not require multi-policy members.
- Two policies: fail before building the MuJoCo scene if the profile lacks the multi-policy contract, either actor's TorchScript load fails, its declared shape is wrong, or a dummy forward returns non-finite values.
- Validate each actor with its own declared `(obs_dim, action_dim)`; reject unknown active policy names and invalid/non-finite runtime observations/actions with step and policy context.
- Existing key events continue to reach `handle_key` whether or not a second policy is present. `handle_policy_key` is called only when a second checkpoint is present, so omitting it makes policy-state transitions unavailable while leaving existing one-policy key behavior unchanged. Reset remains supported and resets profile state and its policy action history.

## State ownership and Chocolate handoff

The generic inference package exposes policy routing and optional transition-control hooks only. The Chocolate profile owns motion loading, the velocity-zero-command stand state, alignment to motion frame zero, ready/hold behavior, the play key, motion-time advancement, and end-of-clip behavior. The selected AMP export is supplied as the primary checkpoint at runtime; no weight is copied into the inference package.

The initial Chocolate key mapping will preserve the existing `R` reset behavior, use `T` to align to the first motion frame, and `P` to begin playback from frame zero. Alignment uses a configurable 1.0 second default; while ready, the profile holds frame zero until `P`. Playback advances at the motion's native timestep, does not loop, and holds the final frame until `R` resets to velocity stand. These defaults must not affect single-policy users.

## Error handling

- `--second-checkpoint` without a multi-policy-capable profile: clear startup error naming the missing contract.
- Missing/invalid checkpoint, mismatched dimensions, non-finite model output, invalid policy key, or invalid runtime observation: fail with checkpoint/policy/step context before unsafe action application where possible.
- Without a second checkpoint, do not invoke `handle_policy_key`; keep existing profile key handling and output unchanged.

## Testing

- Regression test: an old profile implementing only `Sim2SimProfile` loads and runs the one-policy path without any new members.
- Optional-secondary tests: missing second checkpoint leaves routing disabled; supplying it requires the multi-policy extension.
- Validate two TorchScript actors with different observation dimensions and the same or different action dimensions.
- Validate malformed TorchScript, shape mismatch, non-finite dummy output, invalid policy selection, and non-finite runtime observation handling.
- Test profile-owned transition/ready/play dispatch and that transition steps do not advance tracking time before the play state.
- Run a headless finite-step test with fixture policies; no RL training is part of validation.

## Non-goals

- No Chocolate model, dataset, keyboard mapping, reward, or training code inside the generic inference package.
- No edits to the AMP repository or the parent `robot_lab` Docker/DR02 working changes.
- No new inference backend, critic loading, or policy architecture inference from ordinary training checkpoints.
- No requirement to train either policy.
