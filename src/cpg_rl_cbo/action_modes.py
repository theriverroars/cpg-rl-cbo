from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ActionMode(str, Enum):
    """Supported policy-to-CPG action interfaces."""

    PER_LEG_MU_OMEGA_PHI = "per_leg_mu_omega_phi"
    SHARED_OMEGA_PHASE_OFFSETS = "shared_omega_phase_offsets"


@dataclass(frozen=True)
class CPGCommand:
    mu: tuple[float, float, float, float]
    omega: tuple[float, float, float, float]
    phi: tuple[float, float, float, float]


class ActionDecoder:
    """Decodes raw policy outputs to a normalized CPG command for 4 legs."""

    def __init__(self, mode: ActionMode):
        self.mode = mode

    def decode(self, action: Sequence[float]) -> CPGCommand:
        values = tuple(float(v) for v in action)
        if self.mode is ActionMode.PER_LEG_MU_OMEGA_PHI:
            if len(values) != 12:
                raise ValueError("PER_LEG_MU_OMEGA_PHI expects 12 values")
            mu = values[0:4]
            omega = values[4:8]
            phi = values[8:12]
            return CPGCommand(mu=mu, omega=omega, phi=phi)

        if self.mode is ActionMode.SHARED_OMEGA_PHASE_OFFSETS:
            if len(values) != 10:
                raise ValueError("SHARED_OMEGA_PHASE_OFFSETS expects 10 values")
            mu = values[0:4]
            omega_shared = values[4]
            phase_offsets = values[5:9]
            phi_shared = values[9]
            omega = (omega_shared,) * 4
            phi = tuple(phi_shared + offset for offset in phase_offsets)
            return CPGCommand(mu=mu, omega=omega, phi=phi)

        raise ValueError(f"Unsupported action mode: {self.mode}")
