"""Backend registry and factory."""

from __future__ import annotations

from typing import Any

from sim2sim.backends.base import SimulatorBackend

_BACKENDS: dict[str, type[SimulatorBackend]] = {}


def register_backend(name: str, backend_cls: type[SimulatorBackend]) -> None:
    """Register a simulator backend.

    Args:
        name: Backend name (e.g., 'mujoco', 'isaaclab').
        backend_cls: Backend class to register.
    """
    _BACKENDS[name] = backend_cls


def create_backend(name: str, **kwargs: Any) -> SimulatorBackend:
    """Create a simulator backend by name.

    Args:
        name: Registered backend name.
        **kwargs: Arguments passed to the backend constructor.

    Returns:
        Instantiated backend.

    Raises:
        ValueError: If the backend name is not registered.
    """
    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
    return _BACKENDS[name](**kwargs)


def list_backends() -> list[str]:
    """Return list of registered backend names."""
    return sorted(_BACKENDS.keys())


def _register_builtin_backends() -> None:
    """Register built-in backends if their dependencies are available."""
    try:
        from sim2sim.backends.mujoco import MuJoCoBackend

        register_backend("mujoco", MuJoCoBackend)
    except ImportError:
        pass


_register_builtin_backends()
