"""Inner-only postprocessing selection using caches from the SAME outer-fold model.

Never repurpose other folds' OOF predictions as this model's inner validation:
those other models may have trained on the current outer held-out specimens.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .data import load_manifest, load_field
from .postprocess import Postprocess, watershed_labels
from .measurement import measure
from .io import write_json, write_csv, fresh_output, sha256


def tune(inner_manifest: Path, heldout_manifest: Path, output: Path, config: dict) -> dict:
    inner = load_manifest(inner_manifest, require_probability=True)
    heldout = load_manifest(heldout_manifest, require_probability=True)
    if any(r["evaluation_role"] != "inner_validation" or r["provenance_status"] != "checkpoint_verified" for r in inner):
        raise ValueError("Selection requires verified inner_validation cache records.")
    if any(r["evaluation_role"] != "held_out" or r["provenance_status"] != "checkpoint_verified" for r in heldout):
        raise ValueError("Selection target requires verified held_out cache records.")
    if {(r["model_id"], r["run_id"]) for r in inner} != {(r["model_id"], r["run_id"]) for r in heldout}:
        raise ValueError("Inner and held-out caches must refer to the same model/run.")
    outer_folds = sorted({r["model_outer_fold"] for r in heldout})
    if set(outer_folds) != {r["model_outer_fold"] for r in inner}:
        raise ValueError("Inner/heldout outer model folds differ.")
    # Metadata and hashes may be read from held-out records; held-out labels are NOT
    # scored or used for selection. Every fold must use exactly the same checkpoint.
    for fold in outer_folds:
        ir = [r for r in inner if r["model_outer_fold"] == fold]
        hr = [r for r in heldout if r["model_outer_fold"] == fold]
        if {r["group_id"] for r in ir} & {r["group_id"] for r in hr}:
            raise ValueError(f"Inner/heldout specimen overlap for model fold {fold}")
        hashes = {json.loads(r["provenance_path"].read_text(encoding="utf-8"))["checkpoint_sha256"] for r in ir + hr}
        if len(hashes) != 1:
            raise ValueError(f"Inner and heldout probabilities use different checkpoints: model fold {fold}")
    fresh_output(output)
    protocol = config["measurement_protocol"]
    refparams = Postprocess(**config["reference_postprocessing"])
    base = config["prediction_postprocessing"].copy()
    scores, picks = [], {}
    for fold in outer_folds:
        ir = [r for r in inner if r["model_outer_fold"] == fold]
        candidates = {float(t): [] for t in config["thresholds"]}
        for r in ir:
            roi, ref, _, prob = load_field(r, refparams)
            true = measure(ref, roi, r["um_per_pixel"], protocol)
            if true["astm_g"] is None:
                continue  # Same reference eligibility for EVERY candidate.
            for threshold in candidates:
                params = Postprocess(**{**base, "boundary_threshold": threshold})
                pred = measure(watershed_labels(prob, roi, params), roi, r["um_per_pixel"], protocol)
                candidates[threshold].append(abs(pred["astm_g"] - true["astm_g"]) if pred["astm_g"] is not None else None)
        ranks = []
        for threshold, vals in candidates.items():
            defined = [v for v in vals if v is not None]
            failure_rate = (len(vals) - len(defined)) / len(vals) if vals else 1.0
            mae = float(np.mean(defined)) if defined else float("inf")
            scores.append({"model_outer_fold": fold, "threshold": threshold, "n_reference_eligible": len(vals),
                           "n_prediction_defined": len(defined), "prediction_failure_rate": failure_rate,
                           "conditional_g_mae": mae if defined else None})
            # Predeclared lexicographic objective; failures cannot win by disappearing.
            ranks.append((failure_rate, mae, abs(threshold - base["boundary_threshold"]), threshold))
        best = min(ranks)
        if not np.isfinite(best[1]):
            raise ValueError(f"No threshold yields any defined G on eligible inner fields: {fold}")
        picks[str(fold)] = {**base, "boundary_threshold": best[-1]}
    result = {"selected_per_outer_fold": picks, "measurement_protocol": protocol,
              "reference_postprocessing": config["reference_postprocessing"],
              "objective": "lexicographic: failure rate, conditional G MAE, distance from baseline, threshold",
              "selection_status": "inner_only_same_outer_model",
              "inner_manifest_sha256": sha256(inner_manifest), "heldout_manifest_sha256": sha256(heldout_manifest),
              "inner_provenance_sha256": {r["sample_id"]: sha256(r["provenance_path"]) for r in inner},
              "heldout_provenance_sha256": {r["sample_id"]: sha256(r["provenance_path"]) for r in heldout},
              "note": "Shared inner set selects both checkpoints and postprocessing; it is not an additional independent validation set. Outer fields are untouched by selection."}
    write_json(output / "selected_parameters.json", result)
    write_csv(output / "inner_selection_scores.csv", scores)
    return result
