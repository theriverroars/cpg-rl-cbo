from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt

from .action_modes import CPGCommand


@dataclass(frozen=True)
class CPGState:
    r: tuple[float, float, float, float]
    theta: tuple[float, float, float, float]
    theta_dot: tuple[float, float, float, float]
    phi: tuple[float, float, float, float]


@dataclass(frozen=True)
class FootTrajectoryPoint:
    x: float
    y: float
    z: float


class QuadrupedCPG:
    """Minimal CPG oscillator + foot trajectory generator."""

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self._theta = [0.0, 0.0, 0.0, 0.0]

    def step(self, command: CPGCommand, step_length: float, clearance: float, height: float) -> tuple[list[FootTrajectoryPoint], CPGState]:
        r = [sqrt(max(0.0, mu)) for mu in command.mu]
        theta_dot = [max(0.0, omega) for omega in command.omega]
        points: list[FootTrajectoryPoint] = []

        for i in range(4):
            self._theta[i] = (self._theta[i] + theta_dot[i] * self.dt + command.phi[i]) % (2 * pi)
            phase = self._theta[i]
            x = step_length * r[i] * cos(phase)
            swing = max(0.0, sin(phase))
            z = -height + clearance * swing
            y = 0.0
            points.append(FootTrajectoryPoint(x=x, y=y, z=z))

        state = CPGState(
            r=tuple(r),
            theta=tuple(self._theta),
            theta_dot=tuple(theta_dot),
            phi=command.phi,
        )
        return points, state
