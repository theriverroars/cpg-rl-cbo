import argparse
import unittest

from cpg_rl_cbo.action_modes import ActionMode
from cpg_rl_cbo.isaaclab import cli_args
from cpg_rl_cbo.morphology import Terrain
from cpg_rl_cbo.observations import ObservationProfile


class CliArgsTests(unittest.TestCase):
    def test_parse_task_spec_overrides(self):
        parser = argparse.ArgumentParser()
        cli_args.add_task_args(parser)
        args = parser.parse_args(
            [
                "--task",
                "multi_morph_train",
                "--action_mode",
                "shared_omega_phase_offsets",
                "--observation_profile",
                "medium",
                "--morphologies",
                "go1,go2",
                "--terrains",
                "flat,slope",
                "--cbo_h",
                "0.45",
                "--num_envs",
                "32",
            ]
        )
        spec = cli_args.parse_task_spec(args)
        self.assertEqual(spec.action_mode, ActionMode.SHARED_OMEGA_PHASE_OFFSETS)
        self.assertEqual(spec.observation_profile, ObservationProfile.MEDIUM)
        self.assertEqual(spec.morphologies, ("go1", "go2"))
        self.assertEqual(spec.terrains, (Terrain.FLAT, Terrain.SLOPE))
        self.assertAlmostEqual(spec.cbo_parameters.h, 0.45)
        self.assertEqual(spec.num_envs, 32)


if __name__ == "__main__":
    unittest.main()
