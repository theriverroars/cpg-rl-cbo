from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MorphologyCondition:
    mass_kg: float
    scale: float
    link_lengths_m: tuple[float, float]
    com_m: tuple[float, float, float]

    def to_feature_vector(self) -> tuple[float, ...]:
        return (self.mass_kg, self.scale, *self.link_lengths_m, *self.com_m)


@dataclass(frozen=True)
class CBOParameters:
    g_c_hind: float
    g_c_front: float
    g_p_hind: float
    g_p_front: float
    x_offset_front: float
    x_offset_hind: float
    d_set_front: float
    d_step_hind: float
    h: float


def default_cbo_parameters() -> CBOParameters:
    return CBOParameters(
        g_c_hind=1.0,
        g_c_front=1.0,
        g_p_hind=1.0,
        g_p_front=1.0,
        x_offset_front=0.0,
        x_offset_hind=0.0,
        d_set_front=0.5,
        d_step_hind=0.5,
        h=0.30,
    )
