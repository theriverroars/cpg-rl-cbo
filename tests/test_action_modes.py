import unittest

from cpg_rl_cbo.action_modes import ActionDecoder, ActionMode


class ActionDecoderTests(unittest.TestCase):
    def test_per_leg_mode_decodes_triplets(self):
        decoder = ActionDecoder(ActionMode.PER_LEG_MU_OMEGA_PHI)
        cmd = decoder.decode([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(cmd.mu, (0.0, 1.0, 2.0, 3.0))
        self.assertEqual(cmd.omega, (4.0, 5.0, 6.0, 7.0))
        self.assertEqual(cmd.phi, (8.0, 9.0, 10.0, 11.0))

    def test_shared_mode_decodes_offsets(self):
        decoder = ActionDecoder(ActionMode.SHARED_OMEGA_PHASE_OFFSETS)
        cmd = decoder.decode([1, 1, 1, 1, 2, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(cmd.omega, (2.0, 2.0, 2.0, 2.0))
        self.assertEqual(cmd.phi, (0.6, 0.7, 0.8, 0.9))


if __name__ == "__main__":
    unittest.main()
