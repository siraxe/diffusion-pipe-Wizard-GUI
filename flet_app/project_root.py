from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory.
    Prefers flet_app.settings.PROJECT_ROOT; falls back to parent of flet_app/.
    """
    try:
        from flet_app.settings import settings  # type: ignore
        pr = settings.get("PROJECT_ROOT")
        if pr:
            return Path(pr)
    except Exception:
        pass
    # This file lives in flet_app/, so its parent is the project root
    return Path(__file__).resolve().parent.parent

