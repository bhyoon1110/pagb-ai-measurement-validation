from __future__ import annotations
import copy
import math
from .corruptions import KINDS
import json
from pathlib import Path
from .postprocess import Postprocess

DEFAULT = {
    "measurement_protocol": "strict_v2",
    "prediction_postprocessing": Postprocess().to_dict(),
    "reference_postprocessing": Postprocess(0.5, 10, 40).to_dict(),
    "iou_thresholds": [0.5, 0.75],
    "boundary_tolerances_px": [0, 1, 2, 4],
    "boundary_tolerances_um": [],
    "thresholds": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "sweep_min_distances": [6, 8, 10, 12],
    "sweep_min_grain_areas": [20, 40, 80],
    "sweep_topology_modes": ["legacy", "none"],
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 20260928,
    "reference_g_bins": [8.0, 10.0],
    "diagnostic_g_tolerances": [0.1, 0.3, 0.5, 1.0],
    "diagnostic_f1_thresholds": [0.25, 0.5, 0.75],
    "corruption_kinds": ["random_selection", "small_first_selection", "large_first_selection", "merge", "split", "merge_to_border"],
    "corruption_strengths": [0.0, 0.10, 0.25, 0.50, 0.75],
    "corruption_repeats": 10,
    "corruption_seed": 20260928,
    "save_matches": True,
    "postprocessing_selection_status": "fixed_exploratory_not_independent_selection"
}


def load_config(path: Path | None = None) -> dict:
    config = copy.deepcopy(DEFAULT)
    if path:
        override = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        unknown = set(override) - set(config)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        for k, v in override.items():
            if isinstance(config[k], dict):
                config[k].update(v)
            else:
                config[k] = v
    Postprocess(**config["prediction_postprocessing"]).validate()
    Postprocess(**config["reference_postprocessing"]).validate()
    if config["measurement_protocol"] not in ("legacy_v1", "strict_v2"):
        raise ValueError("Invalid measurement_protocol.")
    if 0.5 not in config["iou_thresholds"] or any(not 0 < x <= 1 for x in config["iou_thresholds"]):
        raise ValueError("iou_thresholds must include 0.5 and lie in (0,1].")
    if config["bootstrap_replicates"] < 0 or config["corruption_repeats"] < 1:
        raise ValueError("Invalid repetitions.")
    for key in ("thresholds", "corruption_strengths"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must be nonempty and unique.")
    if any(not 0 < t < 1 for t in config["thresholds"]):
        raise ValueError("thresholds must lie in (0,1).")
    if any(not 0 <= s <= 1 for s in config["corruption_strengths"]):
        raise ValueError("corruption_strengths must lie in [0,1].")
    if sorted(set(config["reference_g_bins"])) != config["reference_g_bins"]:
        raise ValueError("reference_g_bins must be strictly increasing.")
    for key in ("bootstrap_replicates", "corruption_repeats", "bootstrap_seed", "corruption_seed"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer.")
    for key in ("boundary_tolerances_px", "boundary_tolerances_um", "diagnostic_g_tolerances"):
        if any(not math.isfinite(x) or x < 0 for x in config[key]):
            raise ValueError(f"{key} must contain finite nonnegative values.")
    if any(not 0 <= x <= 1 for x in config["diagnostic_f1_thresholds"]):
        raise ValueError("Diagnostic F1 cutoffs must lie in [0,1].")
    if not config["corruption_kinds"] or set(config["corruption_kinds"]) - set(KINDS):
        raise ValueError("Unknown or empty corruption_kinds.")
    if any(not math.isfinite(x) for x in config["reference_g_bins"]):
        raise ValueError("reference_g_bins must be finite.")
    for key, field in (("sweep_min_distances", "min_distance"), ("sweep_min_grain_areas", "min_grain_area"),
                       ("sweep_topology_modes", "topology_mode")):
        if len(set(config[key])) != len(config[key]):
            raise ValueError(f"{key} must contain unique values.")
        for value in config[key]:
            params = {**config["prediction_postprocessing"], field: value}
            Postprocess(**params).validate()
    return config
