import unittest

from cpg_rl_cbo.isaaclab.task_registry import get_task_spec, list_tasks
from cpg_rl_cbo.morphology import Terrain


class TaskRegistryTests(unittest.TestCase):
    def test_registry_exposes_defaults(self):
        tasks = list(list_tasks())
        self.assertGreaterEqual(len(tasks), 2)

    def test_task_spec_contents(self):
        spec = get_task_spec("multi_morph_train")
        self.assertIn("go1", spec.morphologies)
        self.assertIn(Terrain.FLAT, spec.terrains)
        self.assertGreater(spec.num_envs, 0)


if __name__ == "__main__":
    unittest.main()
