import unittest

from cpg_rl_cbo.action_modes import ActionMode
from cpg_rl_cbo.cpg import CPGState
from cpg_rl_cbo.observations import ObservationInput, ObservationProfile
from cpg_rl_cbo.pipeline import GaitAdaptationPipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_generates_12_joint_targets(self):
        pipeline = GaitAdaptationPipeline(
            robot_name="go1",
            action_mode=ActionMode.SHARED_OMEGA_PHASE_OFFSETS,
            observation_profile=ObservationProfile.MEDIUM,
        )
        obs = ObservationInput(
            velocity_command=(0.3, 0.0, 0.0),
            orientation_quat=(1.0, 0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            joint_positions=(0.0,) * 12,
            joint_velocities=(0.0,) * 12,
            foot_contacts=(True, True, False, False),
            last_action=(0.0,) * 10,
            cpg_state=CPGState(r=(0, 0, 0, 0), theta=(0, 0, 0, 0), theta_dot=(0, 0, 0, 0), phi=(0, 0, 0, 0)),
        )
        out = pipeline.step((0.5, 0.5, 0.5, 0.5, 2.0, 0.0, 1.57, 3.14, 4.71, 0.0), obs)
        self.assertEqual(len(out.joint_targets), 12)


if __name__ == "__main__":
    unittest.main()
