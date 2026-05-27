from __future__ import annotations

from dataclasses import dataclass

from ..action_modes import ActionMode
from ..cbo import CBOParameters
from ..observations import ObservationInput, ObservationProfile
from ..pipeline import GaitAdaptationPipeline


@dataclass(frozen=True)
class ControllerConfig:
    robot_name: str
    action_mode: ActionMode
    observation_profile: ObservationProfile
    cbo_parameters: CBOParameters


class GaitAdaptationController:
    """Bridge between IsaacLab environments and the RL→CPG→IK pipeline."""

    def __init__(self, config: ControllerConfig):
        self.pipeline = GaitAdaptationPipeline(
            robot_name=config.robot_name,
            action_mode=config.action_mode,
            observation_profile=config.observation_profile,
            cbo_parameters=config.cbo_parameters,
        )

    def compute_joint_targets(
        self, policy_action: tuple[float, ...], observation_input: ObservationInput
    ) -> tuple[list[float], tuple[float, ...]]:
        output = self.pipeline.step(policy_action, observation_input)
        return output.joint_targets, output.cpg_state.theta
