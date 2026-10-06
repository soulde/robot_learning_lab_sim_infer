import threading
import time
from types import SimpleNamespace

import pytest

from robot_learning_lab_sim_infer.configs import RuntimeConfig
from robot_learning_lab_sim_infer.nodes.sim_hardware_node import SimHardwareNode
from robot_learning_lab_sim_infer.runtime.rc import RCValues


def test_cyclone_reliable_qos_builds_with_supported_blocking_duration():
    pytest.importorskip("cyclonedds")
    from robot_learning_lab_sim_infer.dds.transport import make_qos_profiles

    latest, reliable = make_qos_profiles()
    assert latest is not None
    assert reliable is not None


class FakeProfile:
    name = "fake"
    joint_names = ("joint",)
    simulation_dt = 0.001

    def __init__(self):
        self.reset_calls = 0

    def build_model(self):
        return object()

    def reset(self, model, data):
        self.reset_calls += 1
        data.steps = 0
        data.command = None

    def extract_robot_state(self, model, data, timestamp_ns):
        return {"timestamp_ns": timestamp_ns, "steps": data.steps}

    def make_rc_command(self, values, timestamp_ns):
        return {"timestamp_ns": timestamp_ns, "enabled": values.enabled}

    def apply_motor_command(self, model, data, command):
        data.command = command

    def apply_safe_command(self, model, data):
        data.command = None


class FakeDDS:
    def __init__(self):
        self.states = []
        self.rc = []
        self.commands = []

    def take_motor_command(self):
        return self.commands.pop(0) if self.commands else None

    def publish_robot_state(self, state):
        self.states.append(state)

    def publish_rc_command(self, command):
        self.rc.append(command)

    def close(self):
        pass


class FakeRenderer:
    def __init__(self):
        self.frames = 0

    def update(self, model, data):
        self.frames += 1

    def close(self):
        pass


def test_node_runs_sim_render_and_publish_threads_and_routes_command():
    stop = threading.Event()
    data = SimpleNamespace(steps=0, command=None)
    dds = FakeDDS()
    renderer = FakeRenderer()
    dds.commands.append("motor")

    def physics_step(model, data):
        data.steps += 1
        if data.steps >= 10:
            stop.set()

    node = SimHardwareNode(
        FakeProfile(),
        RuntimeConfig(),
        dds,
        renderer,
        data_factory=lambda model: data,
        physics_step=physics_step,
    )
    node.run(stop)

    assert data.steps >= 10
    assert data.command == "motor"
    assert renderer.frames > 0
    assert dds.states
    assert dds.rc
    assert {thread.name for thread in node._threads} == {
        "sim-hardware-simulation", "sim-hardware-render", "sim-hardware-dds"
    }


def test_node_copies_latest_rc_values_for_publication():
    stop = threading.Event()
    data = SimpleNamespace(steps=0, command=None)
    dds = FakeDDS()

    def physics_step(model, data):
        data.steps += 1
        if data.steps >= 2:
            stop.set()

    node = SimHardwareNode(
        FakeProfile(), RuntimeConfig(), dds, FakeRenderer(),
        data_factory=lambda model: data, physics_step=physics_step,
    )
    node.set_rc_values(RCValues(enabled=True, vx=0.4))
    node.run(stop)
    assert any(sample["enabled"] for sample in dds.rc)


def test_reset_request_from_renderer_is_applied_by_simulation_worker():
    stop = threading.Event()
    data = SimpleNamespace(steps=0, command="stale")
    dds = FakeDDS()
    callbacks = {}

    def renderer_factory(model, on_rc_update, on_reset):
        callbacks["reset"] = on_reset
        on_reset()
        return FakeRenderer()

    def physics_step(model, data):
        data.steps += 1
        if data.steps >= 3:
            stop.set()

    profile = FakeProfile()
    node = SimHardwareNode(
        profile, RuntimeConfig(), dds, renderer_factory,
        data_factory=lambda model: data, physics_step=physics_step,
    )
    node.run(stop)

    assert profile.reset_calls == 2
    assert callbacks["reset"] is not None
    assert data.steps >= 3
    assert data.command is None
