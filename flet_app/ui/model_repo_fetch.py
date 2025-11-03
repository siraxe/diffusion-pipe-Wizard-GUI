import json
import re
from pathlib import Path
from flet_app.project_root import get_project_root
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


HF_TREE_API = "https://huggingface.co/api/models/{org}/{repo}/tree/{branch}?recursive=1"
HF_RESOLVE_URL = "https://huggingface.co/{org}/{repo}/resolve/{branch}/{path}"


def get_models_root() -> Path:
    """Return the repo's models/ folder path under project root."""
    return get_project_root() / "models"


def parse_hf_repo(input_str: str) -> dict:
    """
    Parse user input into (org, repo, branch) and a normalized tree URL.
    Accepts one of:
      - https://huggingface.co/ORG/REPO
      - https://huggingface.co/ORG/REPO/tree/BRANCH
      - https://huggingface.co/ORG/REPO/tree/BRANCH/sub/path
      - ORG/REPO
    Defaults branch to 'main' when absent.
    Returns dict: {org, repo, branch, subpath, tree_url}
    Raises ValueError on invalid input.
    """
    s = input_str.strip()
    if not s:
        raise ValueError("Empty input string")

    # Full URL with optional /tree/<branch> and optional subpath following branch
    # Examples:
    #  - https://huggingface.co/org/repo
    #  - https://huggingface.co/org/repo/tree/main
    #  - https://huggingface.co/org/repo/tree/main/sub/dir
    m = re.match(r"^https?://huggingface\.co/([^/]+)/([^/]+)(?:/tree/([^/?#]+)(?:/(.*))?)?", s)
    if m:
        org, repo = m.group(1), m.group(2)
        branch = (m.group(3) or "main")
        subpath = (m.group(4) or "").strip()
        # Normalize tree URL for display (include subpath if present)
        tree_url = f"https://huggingface.co/{org}/{repo}/tree/{branch}"
        if subpath:
            tree_url = f"{tree_url}/{subpath}"
        return {"org": org, "repo": repo, "branch": branch, "subpath": subpath, "tree_url": tree_url}

    # Shorthand ORG/REPO
    m2 = re.match(r"^([^/]+)/([^/]+)$", s)
    if m2:
        org, repo, branch = m2.group(1), m2.group(2), "main"
        tree_url = f"https://huggingface.co/{org}/{repo}/tree/{branch}"
        return {"org": org, "repo": repo, "branch": branch, "subpath": "", "tree_url": tree_url}

    raise ValueError("Provide a Hugging Face model repo like 'ORG/REPO' or full URL")


def list_hf_file_urls(org: str, repo: str, branch: str, path_prefix: str | None = None) -> list[str]:
    """
    Query Hugging Face API for repo tree and return direct 'resolve' file URLs.
    Uses: /api/models/{org}/{repo}/tree/{branch}?recursive=1
    """
    api_url = HF_TREE_API.format(org=org, repo=repo, branch=branch)
    req = Request(api_url, headers={"User-Agent": "dpipe-gui/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise RuntimeError(f"Hugging Face API error {e.code}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e

    files = []
    # API returns a list of nodes: {"path": ..., "type": "file" | "directory"}
    for node in data:
        node_type = node.get("type") or node.get("type_", "")
        if str(node_type).lower() in ("file", "blob"):
            path = node.get("path")
            if path:
                if path_prefix:
                    # Keep only files under the requested subpath
                    if not (path == path_prefix or path.startswith(path_prefix.rstrip("/") + "/")):
                        continue
                files.append(
                    HF_RESOLVE_URL.format(org=org, repo=repo, branch=branch, path=path)
                )
    return files


def list_hf_file_entries(org: str, repo: str, branch: str, path_prefix: str | None = None) -> list[tuple[str, str]]:
    """
    Return list of (relative_path, resolve_url) entries for files in the repo.
    """
    api_url = HF_TREE_API.format(org=org, repo=repo, branch=branch)
    req = Request(api_url, headers={"User-Agent": "dpipe-gui/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        raise RuntimeError(f"Hugging Face API error {e.code}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e

    entries: list[tuple[str, str]] = []
    for node in data:
        node_type = node.get("type") or node.get("type_", "")
        if str(node_type).lower() in ("file", "blob"):
            path = node.get("path")
            if path:
                if path_prefix:
                    if not (path == path_prefix or path.startswith(path_prefix.rstrip("/") + "/")):
                        continue
                url = HF_RESOLVE_URL.format(org=org, repo=repo, branch=branch, path=path)
                entries.append((path, url))
    return entries


def ensure_target_dir(subfolder: str | None, repo_name: str) -> Path:
    """
    Determine target folder under models/. If subfolder is empty, use repo name.
    If exists, reuse it (for gap-filling); otherwise, create it and return the Path.
    """
    models_root = get_models_root()
    models_root.mkdir(parents=True, exist_ok=True)
    folder_name = (subfolder or repo_name).strip()
    if not folder_name:
        raise ValueError("Target subfolder cannot be empty")
    target = models_root / folder_name
    if not target.exists():
        target.mkdir(parents=True, exist_ok=False)
    return target


def prepare_hf_file_list(input_str: str, subfolder: str | None = None) -> tuple[str, Path, list[str]]:
    """
    High-level helper: parse input, ensure target folder, and list direct file URLs.
    Returns (tree_url, target_dir_path, url_list).
    """
    parsed = parse_hf_repo(input_str)
    org, repo, branch = parsed["org"], parsed["repo"], parsed["branch"]
    subpath = parsed.get("subpath", "")
    tree_url = parsed["tree_url"]
    target_dir = ensure_target_dir(subfolder, repo)
    urls = list_hf_file_urls(org, repo, branch, subpath or None)
    return tree_url, target_dir, urls


def prepare_hf_file_entries(input_str: str, subfolder: str | None = None) -> tuple[str, Path, list[tuple[str, str]]]:
    """
    Parse, ensure target folder, and return (tree_url, target_dir, [(rel_path, url), ...]).
    """
    parsed = parse_hf_repo(input_str)
    org, repo, branch = parsed["org"], parsed["repo"], parsed["branch"]
    subpath = parsed.get("subpath", "")
    tree_url = parsed["tree_url"]
    target_dir = ensure_target_dir(subfolder, repo)
    entries = list_hf_file_entries(org, repo, branch, subpath or None)
    return tree_url, target_dir, entries
