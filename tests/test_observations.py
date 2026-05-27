import unittest

from cpg_rl_cbo.cpg import CPGState
from cpg_rl_cbo.observations import ObservationBuilder, ObservationInput, ObservationProfile


class ObservationBuilderTests(unittest.TestCase):
    def _input(self):
        state = CPGState(r=(1, 1, 1, 1), theta=(0, 0, 0, 0), theta_dot=(2, 2, 2, 2), phi=(3, 3, 3, 3))
        return ObservationInput(
            velocity_command=(0, 0, 0),
            orientation_quat=(1, 0, 0, 0),
            linear_velocity=(0, 0, 0),
            angular_velocity=(0, 0, 0),
            joint_positions=(0,) * 12,
            joint_velocities=(0,) * 12,
            foot_contacts=(True, False, True, False),
            last_action=(0,) * 10,
            cpg_state=state,
        )

    def test_full_is_larger_than_medium(self):
        data = self._input()
        full = ObservationBuilder(ObservationProfile.FULL).build(data)
        medium = ObservationBuilder(ObservationProfile.MEDIUM).build(data)
        self.assertGreater(len(full), len(medium))


if __name__ == "__main__":
    unittest.main()
