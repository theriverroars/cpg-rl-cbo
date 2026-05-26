from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Terrain(str, Enum):
    FLAT = "flat"
    LITTLE_ROUGH = "little_rough"
    SLOPE = "slope"


@dataclass(frozen=True)
class MorphologySpec:
    name: str
    height_cm: float
    step_length_cm: float
    clearance_cm: float
    penetration_cm: float
    x_offset_cm: float
    dof: int
    mass_kg: float
    kp: float
    kd: float
    scale: float
    link_lengths_m: tuple[float, float]
    com_m: tuple[float, float, float]


MORPHOLOGIES: dict[str, MorphologySpec] = {
    "go1": MorphologySpec("go1", 30.0, 13.0, 7.0, 1.0, 0.0, 12, 12.0, 100.0, 2.7, 1.0, (0.213, 0.213), (0.0, 0.0, 0.0)),
    "go2": MorphologySpec("go2", 30.0, 13.0, 7.0, 1.0, 0.0, 12, 15.0, 100.0, 2.7, 1.0, (0.213, 0.213), (0.0, 0.0, 0.0)),
    "a1": MorphologySpec("a1", 30.0, 13.0, 7.0, 1.0, 0.0, 12, 12.0, 100.0, 2.7, 1.0, (0.20, 0.20), (0.0, 0.0, 0.0)),
    "anymal-b": MorphologySpec("anymal-b", 48.0, 17.0, 7.0, 0.0, 10.0, 12, 30.0, 430.0, 20.7, 1.3, (0.28, 0.28), (0.0, 0.0, 0.0)),
}


def get_morphology(name: str) -> MorphologySpec:
    try:
        return MORPHOLOGIES[name.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(MORPHOLOGIES))
        raise ValueError(f"Unsupported morphology '{name}'. Supported: {supported}") from exc
