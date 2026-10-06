"""Cyclone DDS integration and the package's OMG IDL wire contract."""

from pathlib import Path

IDL_PATH = Path(__file__).with_name("idl") / "robot_v1.idl"


def require_cyclonedds() -> None:
    try:
        import cyclonedds  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Cyclone DDS is required for node communication. Install the 'dds' extra "
            "to use the sim hardware and policy nodes."
        ) from exc


def load_message_types():
    """Load the Cyclone ``IdlStruct`` classes matching the bundled IDL."""
    try:
        from .types import load_message_types as load
    except ImportError as exc:
        raise RuntimeError(
            "Cyclone DDS message types are unavailable. Install the 'dds' extra."
        ) from exc
    return load()
