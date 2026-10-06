"""Translate keyboard taps and GUI values into the hardware-facing RC payload."""

from __future__ import annotations

from dataclasses import dataclass

from ..configs import RCConfig
from ..configs.runtime import AxisBinding


@dataclass(frozen=True)
class RCValues:
    enabled: bool = False
    mode: int = 0
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    button_mask: int = 0


class RCMapper:
    """Keep a clamped command; every configured key tap adds one step."""

    _AXES = ("vx", "vy", "yaw_rate")

    def __init__(self, config: RCConfig):
        self._config = config
        for axis, bounds in config.axis_ranges.items():
            if len(bounds) != 2 or bounds[0] > 0 or bounds[1] < 0 or bounds[0] > bounds[1]:
                raise ValueError(f"RC range for {axis!r} must be ordered and include zero")
        self._axes = {axis: 0.0 for axis in self._AXES}
        self._enabled = True
        self._mode = 0
        self._button_mask = 0

    def _range(self, axis: str) -> tuple[float, float]:
        bounds = self._config.axis_ranges.get(
            axis,
            (-abs(float(self._config.scales.get(axis, 1.0))),
             abs(float(self._config.scales.get(axis, 1.0)))),
        )
        return float(bounds[0]), float(bounds[1])

    def _values(self) -> RCValues:
        return RCValues(
            enabled=self._enabled,
            mode=self._mode,
            vx=self._axes["vx"],
            vy=self._axes["vy"],
            yaw_rate=self._axes["yaw_rate"],
            button_mask=self._button_mask,
        )

    def set_values(self, values: RCValues) -> RCValues:
        self._enabled = bool(values.enabled)
        self._mode = int(values.mode)
        for axis in self._AXES:
            low, high = self._range(axis)
            self._axes[axis] = max(low, min(high, float(getattr(values, axis))))
        return self._values()

    def reset(self) -> RCValues:
        """Clear velocity commands while keeping the currently selected mode."""
        self._axes = {axis: 0.0 for axis in self._AXES}
        self._enabled = True
        self._button_mask = 0
        return self._values()

    def key_event(self, key: str, pressed: bool = True) -> RCValues:
        if not pressed:
            return self._values()
        normalized = key.upper()
        self._enabled = True
        self._button_mask = 0
        if normalized == self._config.clear_key.upper():
            self._axes = {axis: 0.0 for axis in self._AXES}
        elif normalized in self._config.mode_keys:
            self._mode = int(self._config.mode_keys[normalized])
        elif normalized in self._config.button_keys:
            bit = int(self._config.button_keys[normalized])
            if bit <= 0 or bit & (bit - 1):
                raise ValueError(f"RC button mapping for {normalized!r} must be a single-bit mask")
            self._button_mask = bit
        else:
            for bound_key, binding in self._config.key_axes.items():
                if bound_key.upper() != normalized:
                    continue
                if isinstance(binding, AxisBinding):
                    axis, direction = binding.axis, binding.direction
                else:
                    axis, direction = binding
                if axis not in self._axes:
                    raise ValueError(f"Unsupported RC axis '{axis}'")
                low, high = self._range(axis)
                step = float(self._config.key_steps.get(axis, self._config.scales.get(axis, 0.1)))
                self._axes[axis] = max(low, min(high, self._axes[axis] + float(direction) * step))
                break
        return self._values()
