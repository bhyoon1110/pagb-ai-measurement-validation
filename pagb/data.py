"""Manifest adapters. Specimen IDs are required, never guessed from a filename.

Canonical paths are relative to the canonical manifest, unless --data-root is set.
Legacy paths are resolved against the explicitly supplied original workspace.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import os
import numpy as np
from .io import read_csv, write_csv, read_array, read_mask, read_probability, resolve_path, sha256
from .measurement import validate_labels
from .postprocess import Postprocess, watershed_labels

PATH_COLUMNS = ("roi_path", "reference_labels_path", "reference_boundary_path", "prediction_labels_path",
                "probability_path", "image_path", "provenance_path")
REQUIRED = ("sample_id", "group_id", "fold", "model_id", "run_id", "evaluation_role", "roi_path")


def load_manifest(path: Path, data_root: Path | None = None,
                  allow_unverified: bool = False, require_probability: bool = False,
                  allow_missing_scale: bool = False) -> list[dict]:
    path = Path(path).resolve()
    rows = read_csv(path)
    if not rows:
        raise ValueError("Manifest has no fields.")
    base = Path(data_root).resolve() if data_root else path.parent
    seen, groups = set(), {}
    model_runs = set()
    for row in rows:
        for key in REQUIRED:
            if not row.get(key, "").strip():
                raise ValueError(f"Manifest is missing {key}: {row.get('sample_id')}")
        sid, group = row["sample_id"], row["group_id"]
        if sid in seen:
            raise ValueError(f"Duplicate sample_id: {sid}. Use one model/run per manifest.")
        if "/" in sid or "\\" in sid or sid in (".", ".."):
            raise ValueError(f"sample_id cannot contain path separators: {sid}")
        seen.add(sid)
        row["fold"] = int(row["fold"])
        if row["fold"] < 0:
            raise ValueError(f"Negative fold: {sid}")
        if group in groups and groups[group] != row["fold"]:
            raise ValueError(f"Specimen leakage: group {group} appears in folds {groups[group]} and {row['fold']}")
        groups[group] = row["fold"]
        model_runs.add((row["model_id"], row["run_id"]))
        row["model_outer_fold"] = int(row.get("model_outer_fold") or row["fold"])
        value = row.get("um_per_pixel", "")
        if value in (None, ""):
            if not allow_missing_scale:
                raise ValueError(f"Missing physical scale: {sid}. Supply calibration or --allow-missing-scale.")
            row["um_per_pixel"] = None
        else:
            row["um_per_pixel"] = float(value)
            if not math.isfinite(row["um_per_pixel"]) or row["um_per_pixel"] <= 0:
                raise ValueError(f"Invalid um_per_pixel for {sid}: {value}")
        mag = row.get("magnification_x", "")
        row["magnification_x"] = int(mag) if mag not in (None, "") else None
        for key in PATH_COLUMNS:
            if row.get(key):
                row[key] = resolve_path(row[key], base)
                if not row[key].is_file():
                    raise FileNotFoundError(f"{sid}: missing {key}: {row[key]}")
            else:
                row[key] = None
        if not (row["reference_labels_path"] or row["reference_boundary_path"]):
            raise ValueError(f"{sid}: reference labels or a reference boundary map is required.")
        if not (row["prediction_labels_path"] or row["probability_path"]):
            raise ValueError(f"{sid}: prediction labels or a probability map is required. Missing != zero prediction.")
        if require_probability and not row["probability_path"]:
            raise ValueError(f"{sid}: probability maps are required for a postprocessing sweep.")
        status = row.get("provenance_status", "unverified")
        if status == "synthetic_demo":
            if row["evaluation_role"] != "synthetic_demo":
                raise ValueError("Synthetic provenance must also have evaluation_role=synthetic_demo.")
        elif status == "checkpoint_verified":
            _verify_prediction_provenance(row)
        elif not allow_unverified:
            raise ValueError(f"{sid}: checkpoint/split provenance is unverified. Use --allow-unverified-provenance only for a clearly labelled exploratory replay.")
        row["provenance_status"] = status
        row["roi_origin"] = row.get("roi_origin") or "unspecified_must_be_reported"
    if len(model_runs) != 1:
        raise ValueError("One model_id/run_id is required per manifest; never silently pool protocols/models.")
    return rows


def _verify_prediction_provenance(row: dict) -> None:
    p = row.get("provenance_path")
    if not p:
        raise ValueError("checkpoint_verified requires a provenance JSON file.")
    if not row.get("reference_labels_path"):
        raise ValueError("Verified neural caches require the exact cached reference label array.")
    record = json.loads(p.read_text(encoding="utf-8"))
    if record.get("sample_id") != row["sample_id"] or record.get("group_id") != row["group_id"]:
        raise ValueError(f"Provenance identity mismatch: {row['sample_id']}")
    if record.get("model_id") != row["model_id"] or record.get("run_id") != row["run_id"]:
        raise ValueError("Provenance model/run mismatch.")
    if record.get("evaluation_role") != row["evaluation_role"]:
        raise ValueError("Provenance evaluation_role mismatch.")
    if int(record["model_outer_fold"]) != row["model_outer_fold"]:
        raise ValueError("Provenance model fold mismatch.")
    recorded_scale = record.get("calibration_um_per_pixel")
    if recorded_scale is not None and (row["um_per_pixel"] is None or
            not math.isclose(float(recorded_scale), row["um_per_pixel"], rel_tol=1e-12)):
        raise ValueError("Physical calibration changed after prediction cache generation.")
    row["calibration_provenance_status"] = "recorded_and_checked" if recorded_scale is not None else "not_recorded_legacy_cache"
    if not record.get("checkpoint_sha256"):
        raise ValueError("Checkpoint hash missing from provenance.")
    train = set(record.get("training_group_ids", []))
    inner = set(record.get("inner_validation_group_ids", []))
    heldout = set(record.get("held_out_group_ids", []))
    if not train or not inner or not heldout or train & inner or train & heldout or inner & heldout:
        raise ValueError("Missing/overlapping checkpoint group lists.")
    if record.get("early_stopping_on") != "inner_validation":
        raise ValueError("Verified cache cannot use outer/legacy early stopping.")
    expected = heldout if row["evaluation_role"] == "held_out" else inner
    if row["evaluation_role"] not in ("held_out", "inner_validation") or row["group_id"] not in expected:
        raise ValueError("Field group is not in the recorded evaluation split.")
    if row["evaluation_role"] == "held_out" and row["fold"] != row["model_outer_fold"]:
        raise ValueError("Held-out field is being predicted by the wrong outer-fold model.")
    for key, hash_key in (("probability_path", "probability_sha256"), ("roi_path", "roi_sha256"),
                          ("reference_labels_path", "reference_labels_sha256")):
        if row.get(key):
            if not record.get(hash_key) or sha256(row[key]) != record[hash_key]:
                raise ValueError(f"Cached {key} changed or is not covered by provenance: {row['sample_id']}")
    # The baseline prefers supplied instance labels over probabilities; therefore
    # those labels must also be bound, or a cache could swap the scored prediction.
    if row.get("prediction_labels_path"):
        if sha256(row["prediction_labels_path"]) != record.get("prediction_labels_sha256"):
            raise ValueError("Prediction labels are not bound to the verified provenance.")


def load_field(row: dict, reference_params: Postprocess) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    roi = read_mask(row["roi_path"])
    if row["reference_labels_path"]:
        reference = validate_labels(read_array(row["reference_labels_path"]), roi)
        if row.get("provenance_status") == "checkpoint_verified":
            record = json.loads(row["provenance_path"].read_text(encoding="utf-8"))
            if record.get("reference_postprocessing") != reference_params.to_dict():
                raise ValueError("Cached reference postprocessing differs from the requested frozen reference protocol.")
    else:
        boundary = read_mask(row["reference_boundary_path"])
        reference = watershed_labels(boundary.astype(np.float32), roi, reference_params)
    pred = validate_labels(read_array(row["prediction_labels_path"]), roi) if row["prediction_labels_path"] else None
    probability = read_probability(row["probability_path"]) if row["probability_path"] else None
    if probability is not None and probability.shape != roi.shape:
        raise ValueError(f"Probability/ROI shape mismatch: {row['sample_id']}")
    return roi, reference, pred, probability


def import_legacy(workspace: Path, output: Path, manifest: Path | None = None,
                  folds: Path | None = None, calibration: Path | None = None,
                  prediction_root: Path | None = None, instance_dir: Path | None = None,
                  model_id: str = "unet", run_id: str = "legacy_unverified") -> dict:
    workspace = Path(workspace).resolve()
    manifest = manifest or workspace / "pipeline/prepared_data/dataset_manifest.csv"
    folds = folds or workspace / "pipeline/prepared_data/folds.csv"
    calibration = calibration or workspace / "pipeline/calibration/scale_calibration.csv"
    prediction_root = prediction_root or workspace / "pipeline/results/current"
    raw = read_csv(manifest)
    fold_rows, cal_rows = read_csv(folds), read_csv(calibration)
    def index_unique(rows, name):
        index = {}
        for r in rows:
            if r["sample_id"] in index:
                raise ValueError(f"Duplicate sample in {name}: {r['sample_id']}")
            index[r["sample_id"]] = r
        return index
    fold_index, cal_index = index_unique(fold_rows, "folds"), index_unique(cal_rows, "calibration")
    index_unique(raw, "dataset manifest")
    if set(fold_index) != {r["sample_id"] for r in raw}:
        raise ValueError("Dataset/fold membership differs. Explicitly create a subset manifest rather than silently dropping fields (e.g. x200).")
    output = Path(output).resolve()
    base = output.parent
    canonical, missing = [], []
    for r in raw:
        sid = r["sample_id"]
        if not r.get("group_id") or not r.get("valid_roi_mask_path"):
            raise ValueError(f"{sid}: explicit group_id and valid_roi_mask_path are required.")
        folder = prediction_root / sid
        prob = next((p for p in (folder / "boundary_probability.npy", folder / "boundary_probability.tif") if p.is_file()), None)
        # Prefer recomputing labels from a probability map under explicit settings.
        labels = None
        if instance_dir:
            labels = next((p for p in (instance_dir / f"{sid}.tif", instance_dir / f"{sid}.npy", instance_dir / f"{sid}.tiff", instance_dir / sid / "grain_labels.npy", instance_dir / sid / "grain_labels.tif") if p.is_file()), None)
            prob = None
        elif prob is None:
            labels = next((p for p in (folder / "grain_labels.npy", folder / "grain_labels.tif") if p.is_file()), None)
        ref = next((p for p in (folder / "reference_labels.npy", folder / "reference_labels.tif") if p.is_file()), None)
        paths = {"roi_path": resolve_path(r["valid_roi_mask_path"], workspace),
                 "reference_labels_path": ref,
                 "reference_boundary_path": resolve_path(r["training_mask_path"], workspace) if not ref else None,
                 "prediction_labels_path": labels, "probability_path": prob,
                 "image_path": resolve_path(r["grayscale_path"], workspace) if r.get("grayscale_path") else None}
        for k, p in paths.items():
            if p is not None and not p.is_file():
                missing.append(f"{sid}: {k}: {p}")
        if not labels and not prob:
            missing.append(f"{sid}: no prediction labels/probabilities found")
        scale = cal_index.get(sid, {}).get("um_per_pixel", "")
        if not scale:
            missing.append(f"{sid}: no calibrated um_per_pixel")
        canonical.append({"sample_id": sid, "group_id": r["group_id"], "fold": fold_index[sid]["fold"],
                          "magnification_x": r.get("magnification_x", ""), "um_per_pixel": scale,
                          "model_id": model_id, "run_id": run_id, "evaluation_role": "held_out",
                          "provenance_status": "unverified", "roi_origin": "annotation_defined_legacy",
                          **{k: os.path.relpath(p, base) if p else "" for k, p in paths.items()}})
    if missing:
        raise FileNotFoundError("Legacy import preflight failed; no fabricated manifest written:\n" + "\n".join(missing))
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing manifest: {output}")
    write_csv(output, canonical)
    return {"manifest": str(output), "n_fields": len(canonical), "provenance_status": "unverified",
            "note": "Import does not establish OOF checkpoint provenance; explicitly label exploratory replay."}
