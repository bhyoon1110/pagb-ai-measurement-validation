"""Strict, portable I/O. Missing observations are never fabricated as zero detections."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from typing import Any
import numpy as np


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Required input is absent: {p}. See docs/DATA_REQUIREMENTS.md")
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [clean_json(v) for v in obj]
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_json(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def write_csv(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = fields or list(dict.fromkeys(k for row in rows for k in row))
    if not keys:
        keys = ["no_observations"]
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_json(v) for k, v in row.items()})


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def environment() -> dict:
    versions = {}
    for name in ("numpy", "scipy", "scikit-image", "scikit-learn", "Pillow", "tifffile",
                 "opencv-python-headless", "matplotlib", "torch", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def read_array(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing array: {p}")
    if p.suffix.lower() == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(p)
    else:
        from PIL import Image
        with Image.open(p) as image:
            arr = np.asarray(image).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-channel 2D array, got {arr.shape}: {p}")
    return arr


def read_probability(path: str | Path) -> np.ndarray:
    """Float [0,1], uint8/255, uint16/65535 only; never min-max normalize a map."""
    arr = read_array(path)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported probability encoding {arr.dtype}: {path}")
    if not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1)):
        raise ValueError(f"Probability must be finite in [0,1]: {path}")
    return arr


def read_mask(path: str | Path) -> np.ndarray:
    arr = read_array(path)
    if not np.isfinite(arr).all():
        raise ValueError(f"Mask contains nonfinite values: {path}")
    return arr > 0


def resolve_path(value: str, base: Path) -> Path:
    # Accept Windows separators on Linux/macOS for relative manifests.
    p = Path(value.replace("\\", "/")).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def fresh_output(path: Path, overwrite: bool = False) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output is not empty: {path}. Choose a new run directory.")
    path.mkdir(parents=True, exist_ok=True)
