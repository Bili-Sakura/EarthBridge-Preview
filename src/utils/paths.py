# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Project path utilities. Load paths from paths.env (PROJECT_ROOT + derived paths)."""

from pathlib import Path

_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """Return PROJECT_ROOT from env, paths.env, or infer from this file."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    # 1. From environment
    import os
    root = os.environ.get("PROJECT_ROOT")
    if root:
        _PROJECT_ROOT = Path(root).resolve()
        return _PROJECT_ROOT

    # 2. Load paths.env from typical locations
    for candidate in [
        Path.cwd() / "paths.env",
        Path(__file__).resolve().parents[2] / "paths.env",
    ]:
        if candidate.is_file():
            _load_paths_env(candidate)
            root = os.environ.get("PROJECT_ROOT")
            if root:
                _PROJECT_ROOT = Path(root).resolve()
                return _PROJECT_ROOT

    # 3. Infer from this file: src/utils/paths.py -> project root
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    return _PROJECT_ROOT


def _load_paths_env(path: Path) -> None:
    """Parse paths.env and set PROJECT_ROOT if not already set."""
    import os
    import re
    if os.environ.get("PROJECT_ROOT"):
        return
    try:
        text = path.read_text()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^PROJECT_ROOT=(.+)$", line)
            if m:
                val = m.group(1).strip().strip('"\'')
                # Skip bash variable expansion (e.g. ${PROJECT_ROOT:-...})
                if "${" in val or "$(" in val:
                    # paths.env lives in project root; use its parent dir
                    os.environ["PROJECT_ROOT"] = str(path.parent.resolve())
                    return
                # Resolve relative to paths.env parent
                p = Path(val)
                if not p.is_absolute():
                    val = str((path.parent / val).resolve())
                else:
                    val = str(p.resolve())
                os.environ["PROJECT_ROOT"] = val
                return
        # No simple PROJECT_ROOT= found; use paths.env parent
        os.environ["PROJECT_ROOT"] = str(path.parent.resolve())
    except OSError:
        pass


def path_from_root(*parts: str) -> Path:
    """Return PROJECT_ROOT / part1 / part2 / ..."""
    return get_project_root() / Path(*parts)
