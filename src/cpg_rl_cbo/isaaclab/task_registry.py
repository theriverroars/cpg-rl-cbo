from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..action_modes import ActionMode
from ..cbo import CBOParameters, default_cbo_parameters
from ..morphology import Terrain
from ..observations import ObservationProfile


@dataclass(frozen=True)
class TaskSpec:
    name: str
    isaaclab_task: str
    description: str
    morphologies: tuple[str, ...]
    terrains: tuple[Terrain, ...]
    action_mode: ActionMode
    observation_profile: ObservationProfile
    cbo_parameters: CBOParameters
    num_envs: int


TASK_REGISTRY: dict[str, TaskSpec] = {
    "multi_morph_train": TaskSpec(
        name="multi_morph_train",
        isaaclab_task="CPGRLCBO-MultiMorph-Train",
        description="Multi-morphology curriculum across flat, little rough, and slope terrains.",
        morphologies=("go1", "go2", "a1", "anymal-b"),
        terrains=(Terrain.FLAT, Terrain.LITTLE_ROUGH, Terrain.SLOPE),
        action_mode=ActionMode.PER_LEG_MU_OMEGA_PHI,
        observation_profile=ObservationProfile.FULL,
        cbo_parameters=default_cbo_parameters(),
        num_envs=4096,
    ),
    "multi_morph_eval": TaskSpec(
        name="multi_morph_eval",
        isaaclab_task="CPGRLCBO-MultiMorph-Eval",
        description="Evaluation task for multi-morphology gaits on mixed terrains.",
        morphologies=("go1", "go2", "a1", "anymal-b"),
        terrains=(Terrain.FLAT, Terrain.LITTLE_ROUGH, Terrain.SLOPE),
        action_mode=ActionMode.SHARED_OMEGA_PHASE_OFFSETS,
        observation_profile=ObservationProfile.MEDIUM,
        cbo_parameters=default_cbo_parameters(),
        num_envs=64,
    ),
}


def list_tasks() -> Iterable[TaskSpec]:
    return TASK_REGISTRY.values()


def get_task_spec(name: str) -> TaskSpec:
    key = name.lower()
    if key not in TASK_REGISTRY:
        supported = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(f"Unknown task '{name}'. Supported: {supported}")
    return TASK_REGISTRY[key]
