from __future__ import annotations

import importlib
from types import ModuleType


class IsaacLabUnavailable(RuntimeError):
    """Raised when IsaacLab or required runtime dependencies are missing."""


def require_module(module_name: str, install_hint: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise IsaacLabUnavailable(f"Missing dependency '{module_name}'. {install_hint}") from exc


def require_isaaclab() -> None:
    """Ensure the IsaacLab core package is importable."""
    require_module("omni.isaac.lab", "Install IsaacLab and ensure it is on PYTHONPATH.")
