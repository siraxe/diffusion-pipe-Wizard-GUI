"""
Shared base class for musubi command builders.

Provides common path resolution and config-parsing helpers used by the
per-model Run/Cache handler classes (mmh3, ltx2, wan22).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class CommandBuilder:
    """Base class providing project-root discovery and path/config helpers."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.musubi_root = self.project_root / "diffusion-trainers" / "musubi-tuner"

    @staticmethod
    def _find_project_root() -> Path:
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "flet_app").exists() or (parent / "diffusion-trainers").exists():
                return parent
        return Path.cwd()

    def _resolve_path(self, path: str) -> str:
        if not path:
            return ""

        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p

        return str(p)

    @staticmethod
    def parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value

        if value is None:
            return False

        return str(value).strip().lower() in ("true", "1", "yes", "on")
