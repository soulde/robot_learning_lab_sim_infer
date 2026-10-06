import threading
from types import SimpleNamespace

import numpy as np

from robot_learning_lab_sim_infer.configs import RuntimeConfig, StateMachineConfig
from robot_learning_lab_sim_infer.nodes.policy_node import PolicyNode


class FakePolicyProfile:
    name = "policy-test"
    joint_names = ("left", "right")
    policy_dt = 0.001

    def reset(self):
        pass

    def observe(self, policy_name, state, rc):
        return np.array([state.value, rc.vx], dtype=np.float32)

    def infer(self, policy_name, observation):
        assert policy_name == "walk"
        return observation * 2

    def make_motor_command(self, policy_name, action, timestamp_ns):
        return (policy_name, action.tolist(), timestamp_ns)


class FakePolicyDDS:
    def __init__(self, state, rc):
        self.states = [state]
        self.rcs = [rc]
        self.commands = []
        self.closed = False
        self.stop = None

    def take_robot_state(self):
        return self.states.pop(0) if self.states else None

    def take_rc_command(self):
        return self.rcs.pop(0) if self.rcs else None

    def publish_motor_command(self, command):
        self.commands.append(command)
        self.stop.set()

    def close(self):
        self.closed = True


def test_policy_node_owns_fsm_and_publishes_motor_command():
    state = SimpleNamespace(joint_names=("left", "right"), value=3.0)
    rc = SimpleNamespace(enabled=True, mode=0, button_mask=0, vx=0.2)
    dds = FakePolicyDDS(state, rc)
    stop = threading.Event()
    dds.stop = stop
    runtime = RuntimeConfig(
        state_machine=StateMachineConfig(
            initial_state="idle",
            transitions={"idle": {"enable": "active"}},
            state_policies={"idle": None, "active": "walk"},
        )
    )
    PolicyNode(FakePolicyProfile(), runtime, dds).run(stop)
    assert len(dds.commands) == 1
    assert dds.commands[0][0] == "walk"
    assert np.allclose(dds.commands[0][1], [6.0, 0.4])
    assert dds.closed


def test_policy_node_withholds_command_when_rc_is_missing():
    state = SimpleNamespace(joint_names=("left", "right"), value=1.0)
    dds = FakePolicyDDS(state, None)
    stop = threading.Event()
    timer = threading.Timer(0.01, stop.set)
    timer.start()
    runtime = RuntimeConfig(
        state_machine=StateMachineConfig(
            initial_state="active", state_policies={"active": "walk"}
        )
    )
    PolicyNode(FakePolicyProfile(), runtime, dds).run(stop)
    timer.join()
    assert dds.commands == []
    assert dds.closed
