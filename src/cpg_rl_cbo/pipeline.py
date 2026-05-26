from __future__ import annotations

from dataclasses import dataclass

from .action_modes import ActionDecoder, ActionMode
from .cbo import CBOParameters, default_cbo_parameters
from .cpg import CPGState, QuadrupedCPG
from .ik import InverseKinematicsModel, get_ik_model
from .morphology import MorphologySpec, get_morphology
from .observations import ObservationBuilder, ObservationInput, ObservationProfile


@dataclass
class PipelineOutput:
    observation: tuple[float, ...]
    joint_targets: list[float]
    cpg_state: CPGState


class GaitAdaptationPipeline:
    """RL -> CPG -> IK pipeline with modular action and observation variants."""

    def __init__(
        self,
        robot_name: str,
        action_mode: ActionMode,
        observation_profile: ObservationProfile,
        cbo_parameters: CBOParameters | None = None,
        ik_model: InverseKinematicsModel | None = None,
    ):
        self.morphology: MorphologySpec = get_morphology(robot_name)
        self.cpg = QuadrupedCPG()
        self.action_decoder = ActionDecoder(action_mode)
        self.observation_builder = ObservationBuilder(observation_profile)
        self.cbo_parameters = cbo_parameters or default_cbo_parameters()
        self.ik_model = ik_model or get_ik_model(robot_name)

    def step(self, policy_action: tuple[float, ...], observation_input: ObservationInput) -> PipelineOutput:
        observation = self.observation_builder.build(observation_input)
        command = self.action_decoder.decode(policy_action)
        feet, cpg_state = self.cpg.step(
            command,
            step_length=self.morphology.step_length_cm / 100.0,
            clearance=self.morphology.clearance_cm / 100.0,
            height=self.cbo_parameters.h,
        )
        joint_targets = self.ik_model.solve(feet)
        return PipelineOutput(observation=observation, joint_targets=joint_targets, cpg_state=cpg_state)
