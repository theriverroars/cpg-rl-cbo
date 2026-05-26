from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .cpg import CPGState


class ObservationProfile(str, Enum):
    FULL = "full"
    MEDIUM = "medium"


@dataclass(frozen=True)
class ObservationInput:
    velocity_command: tuple[float, float, float]
    orientation_quat: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    joint_positions: tuple[float, ...]
    joint_velocities: tuple[float, ...]
    foot_contacts: tuple[bool, bool, bool, bool]
    last_action: tuple[float, ...]
    cpg_state: CPGState


class ObservationBuilder:
    def __init__(self, profile: ObservationProfile):
        self.profile = profile

    def build(self, data: ObservationInput) -> tuple[float, ...]:
        body_state = data.orientation_quat + data.linear_velocity + data.angular_velocity
        base = data.velocity_command + body_state + tuple(float(v) for v in data.foot_contacts)
        cpg = data.cpg_state.r + data.cpg_state.theta + data.cpg_state.theta_dot + data.cpg_state.phi

        if self.profile is ObservationProfile.FULL:
            joint_state = data.joint_positions + data.joint_velocities
            return base + joint_state + data.last_action + cpg

        if self.profile is ObservationProfile.MEDIUM:
            return base + cpg

        raise ValueError(f"Unsupported observation profile: {self.profile}")
