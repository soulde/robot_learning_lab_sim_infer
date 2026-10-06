"""Small deterministic state machine for the policy process."""

from __future__ import annotations

from dataclasses import dataclass

from ..configs import StateMachineConfig


@dataclass(frozen=True)
class RuntimeDecision:
    state: str
    policy: str | None


class RuntimeStateMachine:
    """Evaluate named events; this class is intended for policy-node use only."""

    def __init__(self, config: StateMachineConfig):
        self._config = config
        if config.initial_state not in config.state_policies:
            raise ValueError(f"Initial state '{config.initial_state}' has no state_policies entry")
        self._state = config.initial_state
        self._validate()

    def _validate(self) -> None:
        known_states = set(self._config.state_policies)
        for state, transitions in self._config.transitions.items():
            if state not in known_states:
                raise ValueError(f"Transitions reference unknown state '{state}'")
            unknown = set(transitions.values()) - known_states
            if unknown:
                raise ValueError(f"Transitions from '{state}' target unknown states: {', '.join(sorted(unknown))}")

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> RuntimeDecision:
        self._state = self._config.initial_state
        return self.decision

    def update(self, event: str | None) -> RuntimeDecision:
        if event is not None:
            target = self._config.transitions.get(self._state, {}).get(event)
            if target is not None:
                self._state = target
        return self.decision

    @property
    def decision(self) -> RuntimeDecision:
        return RuntimeDecision(self._state, self._config.state_policies[self._state])
