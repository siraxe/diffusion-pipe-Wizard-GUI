"""
TOML formatting and path utilities for configuration.

Provides common functions for:
- Quoting values for TOML output
- Converting values to/from boolean
- Path normalization and expansion
"""

import os
import re
from pathlib import Path
from typing import Any, Optional, Union

from flet_app.project_root import get_project_root as _get_project_root


# =============================================================================
# TOML Value Formatting
# =============================================================================

def quote(value: Any) -> str:
    if value is None:
        return "''"
    return f"'{str(value)}'"


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    return s in ('1', 'true', 'yes', 'on')


def toml_bool(value: Any) -> str:
    """
    Convert a value to TOML boolean string.

    Args:
        value: The value to convert

    Returns:
        'true' or 'false' string for TOML
    """
    return 'true' if to_bool(value) else 'false'


# =============================================================================
# Path Utilities
# =============================================================================

def normalize_slashes(path: str) -> str:
    if not path:
        return path
    return str(path).strip().replace('\\', '/')


def is_absolute_path(path: str) -> bool:
    if not path:
        return False
    return bool(re.match(r"^[A-Za-z]:[\\/]|^/|^\\\\", path))


def get_project_root() -> Path:
    """Get the project root directory."""
    return _get_project_root()


def expand_model_path(path: str) -> str:
    if not path or not isinstance(path, str):
        return path

    path = normalize_slashes(path)

    # If already absolute, return as-is
    if is_absolute_path(path):
        return path

    # Get project root and expand
    project_root = get_project_root()
    expanded_path = os.path.join(str(project_root), path)

    return normalize_slashes(expanded_path)


def collapse_model_path(path: str) -> str:
    if not path or not isinstance(path, str):
        return path

    path = normalize_slashes(path)

    # Get project root
    project_root = get_project_root()
    project_root_str = normalize_slashes(str(project_root)).rstrip('/')

    # If path is under project root, make it relative
    if path.lower().startswith(project_root_str.lower() + '/'):
        relative_path = path[len(project_root_str) + 1:]
        return relative_path

    # If not under project root, return as-is
    return path


def collapse_path_to_relative(path: str) -> str:
    try:
        if isinstance(path, str):
            proj_root = get_project_root()
            od = path.replace('\\', '/')
            proj_root_str = str(proj_root).replace('\\', '/').rstrip('/')
            if od.lower().startswith(proj_root_str.lower() + '/'):
                rel = od[len(proj_root_str.rstrip('/')) + 1:]
                return rel
    except Exception:
        pass
    return path


# =============================================================================
# Path Resolution Helpers
# =============================================================================

def resolve_output_dir(raw_output_dir: str, project_root: Path) -> str:
    raw_output_dir = normalize_slashes(raw_output_dir)

    if is_absolute_path(raw_output_dir):
        resolved = os.path.normpath(raw_output_dir)
    else:
        resolved = os.path.normpath(os.path.join(str(project_root), *raw_output_dir.split('/')))

    return normalize_slashes(resolved)


def resolve_path_if_relative(path: str, project_root: Path) -> str:
    path = normalize_slashes(path)

    if is_absolute_path(path):
        return normalize_slashes(os.path.normpath(path))
    else:
        return normalize_slashes(os.path.normpath(os.path.join(str(project_root), *path.split('/'))))
