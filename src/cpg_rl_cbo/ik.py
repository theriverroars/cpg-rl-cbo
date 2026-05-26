from __future__ import annotations

from abc import ABC, abstractmethod
from math import acos, atan2, sqrt

from .cpg import FootTrajectoryPoint
from .morphology import MorphologySpec, get_morphology


class InverseKinematicsModel(ABC):
    @abstractmethod
    def solve(self, foot_targets: list[FootTrajectoryPoint]) -> list[float]:
        """Return flattened joint targets [leg0_3dof, ..., leg3_3dof]."""


class GenericQuadrupedIK(InverseKinematicsModel):
    """Simple 3-DoF per-leg IK suitable as a shared baseline for go1/go2/a1/anymal-b."""

    def __init__(self, morphology: MorphologySpec):
        self.morphology = morphology
        self.l1, self.l2 = morphology.link_lengths_m

    def solve(self, foot_targets: list[FootTrajectoryPoint]) -> list[float]:
        if len(foot_targets) != 4:
            raise ValueError("Expected 4 foot targets for a quadruped")

        joint_targets: list[float] = []
        for target in foot_targets:
            hip_abduction = atan2(target.y, abs(target.z) + 1e-6)
            x = target.x
            z = -target.z
            dist = sqrt(x * x + z * z)
            cos_knee = (dist * dist - self.l1 * self.l1 - self.l2 * self.l2) / (2.0 * self.l1 * self.l2)
            cos_knee = max(-1.0, min(1.0, cos_knee))
            knee = acos(cos_knee)
            hip = atan2(z, x + 1e-6) - atan2(self.l2 * knee, self.l1 + self.l2 * cos_knee)
            joint_targets.extend([hip_abduction, hip, -knee])
        return joint_targets


def get_ik_model(robot_name: str) -> InverseKinematicsModel:
    morphology = get_morphology(robot_name)
    return GenericQuadrupedIK(morphology)
