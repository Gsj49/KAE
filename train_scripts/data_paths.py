"""Repository-relative defaults shared by the paper entry points."""

import json
import os
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/paths.json"


def data_root():
    root = Path(os.environ.get("KAE_DATA_ROOT", "data")).expanduser()
    return (root if root.is_absolute() else ROOT / root).resolve()


def data_path(*keys):
    value = json.loads(CONFIG.read_text())
    for key in keys:
        value = value[key]
    if not isinstance(value, str):
        raise ValueError(f"Not a data path: {keys}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("data/paths.json must use paths relative to the data root")
    return data_root().joinpath(*relative.parts)
