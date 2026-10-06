"""Cyclone DDS endpoints for the three sim/policy wire topics."""

from __future__ import annotations

import os
from typing import Any

from ..configs import DDSConfig


def make_qos_profiles():
    """Construct the latest-value sensor and bounded-wait command QoS pairs."""
    from cyclonedds.core import Qos
    from cyclonedds.qos import Policy
    from cyclonedds.util import duration

    latest = Qos(Policy.History.KeepLast(1), Policy.Reliability.BestEffort)
    reliable = Qos(
        Policy.History.KeepLast(1),
        Policy.Reliability.Reliable(max_blocking_time=duration(milliseconds=100)),
    )
    return latest, reliable


class SimHardwareDDS:
    """Single-participant DDS I/O used by `sim_hardware_node.py`."""

    def __init__(self, config: DDSConfig):
        try:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.pub import DataWriter
            from cyclonedds.sub import DataReader
            from cyclonedds.topic import Topic
        except ImportError as exc:
            raise RuntimeError("Install robot-learning-lab-sim-infer[dds] to use DDS nodes") from exc
        try:
            from .types import RCCommand, RobotState, MotorCommand
        except ImportError as exc:
            raise RuntimeError(
                "Cyclone DDS message types are unavailable; install the 'dds' extra."
            ) from exc

        # Cyclone reads this process configuration while constructing its first participant.
        if config.cyclonedds_uri:
            os.environ["CYCLONEDDS_URI"] = config.cyclonedds_uri
        self._entities: list[Any] = []
        self.participant = DomainParticipant(domain_id=config.domain_id)
        self._entities.append(self.participant)
        state_topic = Topic(self.participant, config.state_topic, RobotState)
        rc_topic = Topic(self.participant, config.rc_topic, RCCommand)
        motor_topic = Topic(self.participant, config.motor_command_topic, MotorCommand)
        self._entities.extend((state_topic, rc_topic, motor_topic))
        latest, reliable = make_qos_profiles()
        self._state_writer = DataWriter(self.participant, state_topic, qos=latest)
        self._rc_writer = DataWriter(self.participant, rc_topic, qos=latest)
        self._motor_reader = DataReader(self.participant, motor_topic, qos=reliable)
        self._entities.extend((self._state_writer, self._rc_writer, self._motor_reader))

    def take_motor_command(self) -> Any | None:
        """Drain queued samples and return only the newest one."""
        newest = None
        while True:
            sample = self._motor_reader.take_next()
            if sample is None:
                return newest
            newest = sample

    def publish_robot_state(self, sample: Any) -> None:
        self._state_writer.write(sample)

    def publish_rc_command(self, sample: Any) -> None:
        self._rc_writer.write(sample)

    def close(self) -> None:
        self._entities.clear()
        # Cyclone DDS Python entities delete themselves when their last Python
        # reference is released; unlike some clients they have no public close().
        self._state_writer = None
        self._rc_writer = None
        self._motor_reader = None
        self.participant = None


class PolicyDDS:
    """Policy-side latest-sample readers and reliable command writer."""

    def __init__(self, config: DDSConfig):
        try:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.pub import DataWriter
            from cyclonedds.sub import DataReader
            from cyclonedds.topic import Topic
        except ImportError as exc:
            raise RuntimeError("Install robot-learning-lab-sim-infer[dds] to use DDS nodes") from exc
        if config.cyclonedds_uri:
            os.environ["CYCLONEDDS_URI"] = config.cyclonedds_uri
        try:
            from .types import MotorCommand, RCCommand, RobotState
        except ImportError as exc:
            raise RuntimeError(
                "Cyclone DDS message types are unavailable; install the 'dds' extra."
            ) from exc
        self._entities: list[Any] = []
        self.participant = DomainParticipant(domain_id=config.domain_id)
        self._entities.append(self.participant)
        state_topic = Topic(self.participant, config.state_topic, RobotState)
        rc_topic = Topic(self.participant, config.rc_topic, RCCommand)
        motor_topic = Topic(self.participant, config.motor_command_topic, MotorCommand)
        self._entities.extend((state_topic, rc_topic, motor_topic))
        latest, reliable = make_qos_profiles()
        self._state_reader = DataReader(self.participant, state_topic, qos=latest)
        self._rc_reader = DataReader(self.participant, rc_topic, qos=latest)
        self._motor_writer = DataWriter(self.participant, motor_topic, qos=reliable)
        self._entities.extend((self._state_reader, self._rc_reader, self._motor_writer))

    @staticmethod
    def _take_latest(reader: Any) -> Any | None:
        newest = None
        while True:
            sample = reader.take_next()
            if sample is None:
                return newest
            newest = sample

    def take_robot_state(self) -> Any | None:
        return self._take_latest(self._state_reader)

    def take_rc_command(self) -> Any | None:
        return self._take_latest(self._rc_reader)

    def publish_motor_command(self, sample: Any) -> None:
        self._motor_writer.write(sample)

    def close(self) -> None:
        self._entities.clear()
        self._state_reader = None
        self._rc_reader = None
        self._motor_writer = None
        self.participant = None
