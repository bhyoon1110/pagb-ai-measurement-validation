"""Export held-out and same-model inner-validation caches with hashes/split records."""
from __future__ import annotations
from pathlib import Path
import json
import os
import sys
import numpy as np
from .io import read_csv, write_csv, write_json, sha256, fresh_output, resolve_path, read_mask
from .postprocess import Postprocess, watershed_labels


def cache_predictions(workspace: Path, output: Path, model_root: Path, config: dict,
                      manifest: Path | None = None, folds: Path | None = None,
                      calibration: Path | None = None, model_id: str = "unet", run_id: str = "eaai_v2",
                      include_inner: bool = False, only_fold: int | None = None,
                      allow_unverified_checkpoints: bool = False,
                      patch_size: int = 256, stride: int = 192, batch_size: int = 8) -> dict:
    import torch
    from PIL import Image
    src = Path(__file__).resolve().parents[1] / "pipeline/src"
    sys.path.insert(0, str(src))
    from evaluate_pipeline import load_model, infer_full_image
    from train_unet import choose_device
    workspace, output, model_root = Path(workspace).resolve(), Path(output).resolve(), Path(model_root).resolve()
    manifest = manifest or workspace / "pipeline/prepared_data/dataset_manifest.csv"
    folds = folds or workspace / "pipeline/prepared_data/folds.csv"
    calibration = calibration or workspace / "pipeline/calibration/scale_calibration.csv"
    raw, fr, cr = read_csv(manifest), read_csv(folds), read_csv(calibration)
    def index(rows, name):
        result = {}
        for r in rows:
            if r["sample_id"] in result:
                raise ValueError(f"Duplicate sample_id in {name}: {r['sample_id']}")
            result[r["sample_id"]] = r
        return result
    dataset, fold_index, cal = index(raw, "dataset"), index(fr, "folds"), index(cr, "calibration")
    if not raw or set(dataset) != set(fold_index):
        raise ValueError("Dataset and fold manifests must contain exactly the same fields.")
    group_fold = {}
    for sid, r in dataset.items():
        if not r.get("group_id"):
            raise ValueError(f"An explicit specimen group_id is required: {sid}")
        if fold_index[sid].get("group_id") and fold_index[sid]["group_id"] != r["group_id"]:
            raise ValueError(f"Specimen ID differs between field/fold manifests: {sid}")
        f, group = int(fold_index[sid]["fold"]), r["group_id"]
        if group in group_fold and group_fold[group] != f:
            raise ValueError(f"Group leakage across folds: {group}")
        group_fold[group] = f
        if sid not in cal or not cal[sid].get("um_per_pixel") or not np.isfinite(float(cal[sid]["um_per_pixel"])) or float(cal[sid]["um_per_pixel"]) <= 0:
            raise ValueError(f"Missing or invalid per-image calibration: {sid}")
        for key in ("grayscale_path", "training_mask_path", "valid_roi_mask_path"):
            if not r.get(key) or not resolve_path(r[key], workspace).is_file():
                raise FileNotFoundError(f"{sid}: missing {key}: {r.get(key)}")
    fold_ids = sorted(set(group_fold.values()))
    if only_fold is not None:
        if only_fold not in fold_ids:
            raise ValueError(f"Fold {only_fold} is not present.")
        fold_ids = [only_fold]
    checkpoints = {}
    for fold in fold_ids:
        cp = model_root / f"fold_{fold}" / "best_model.pt"
        if not cp.is_file():
            raise FileNotFoundError(f"Missing model: {cp}")
        # Deliberately no fallback to unrestricted pickle deserialization.
        record = torch.load(cp, map_location="cpu", weights_only=True)
        cfg = record.get("config", {})
        outer = cfg.get("held_out_fold", cfg.get("fold"))
        if outer is not None and int(outer) != fold:
            raise ValueError(f"Checkpoint outer fold mismatch: {cp}")
        tr, va, ho = (set(cfg.get(k, [])) for k in ("training_group_ids", "inner_validation_group_ids", "held_out_group_ids"))
        expected = {g for g,f in group_fold.items() if f == fold}
        if tr & expected or va & expected:
            raise ValueError(f"Checkpoint saw held-out specimens during training/selection: {cp}")
        inner_fold = cfg.get("inner_validation_fold")
        verified = bool(tr and va and ho and ho == expected and not (tr & va or tr & ho or va & ho)
                        and inner_fold is not None and int(inner_fold) != fold and cfg.get("protocol_status") == "inner_validation")
        if verified:
            # Reject a field manifest with groups not represented by the checkpoint split.
            if tr | va | ho != set(group_fold):
                raise ValueError(f"Checkpoint group universe differs from evaluation manifest: {cp}")
            if va != {g for g,f in group_fold.items() if f == int(inner_fold)}:
                raise ValueError(f"Checkpoint inner fold membership mismatch: {cp}")
        if not verified and not allow_unverified_checkpoints:
            raise ValueError(f"Checkpoint split provenance incomplete: {cp}. Retrain with updated train_unet.py or explicitly use --allow-unverified-checkpoints for exploratory replay.")
        if include_inner and not verified:
            raise ValueError("Inner-cache export for tuning requires verified checkpoint group records.")
        checkpoints[fold] = (cp, cfg, verified, sha256(cp))
        del record
    fresh_output(output)
    device = choose_device()
    manifests = {"held_out": [], "inner_validation": []}
    seen_inner = set()
    for fold, (checkpoint, cfg, verified, checkpoint_hash) in checkpoints.items():
        model = load_model(checkpoint, device)
        to_score = [(sid, "held_out") for sid,r in dataset.items() if int(fold_index[sid]["fold"]) == fold]
        if include_inner:
            to_score += [(sid, "inner_validation") for sid,r in dataset.items()
                         if r["group_id"] in set(cfg["inner_validation_group_ids"])]
        for sid, role in to_score:
            if role == "inner_validation":
                if sid in seen_inner:
                    raise ValueError("A field is inner validation for multiple outer models. Export one outer fold per cache instead of pooling duplicate sample IDs.")
                seen_inner.add(sid)
            r = dataset[sid]
            folder = output / role / sid
            folder.mkdir(parents=True)
            with Image.open(resolve_path(r["grayscale_path"], workspace)) as im:
                gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
            roi = read_mask(resolve_path(r["valid_roi_mask_path"], workspace))
            boundary = read_mask(resolve_path(r["training_mask_path"], workspace))
            if gray.shape != roi.shape or boundary.shape != roi.shape:
                raise ValueError(f"Shape mismatch: {sid}")
            probability = infer_full_image(model, gray, device, patch_size, stride, batch_size)
            pred = watershed_labels(probability, roi, Postprocess(**config["prediction_postprocessing"]))
            ref = watershed_labels(boundary.astype(np.float32), roi, Postprocess(**config["reference_postprocessing"]))
            paths = {"roi_path": folder / "roi.npy", "reference_labels_path": folder / "reference_labels.npy",
                     "prediction_labels_path": folder / "prediction_labels.npy", "probability_path": folder / "probability.npy"}
            for k, arr in (("roi_path", roi), ("reference_labels_path", ref), ("prediction_labels_path", pred), ("probability_path", probability)):
                np.save(paths[k], arr, allow_pickle=False)
            prov_path = folder / "provenance.json"
            provenance = {"sample_id": sid, "group_id": r["group_id"], "model_id": model_id,
                          "run_id": run_id, "model_outer_fold": fold, "evaluation_role": role,
                          "checkpoint_sha256": checkpoint_hash, "checkpoint_path": str(checkpoint),
                          "early_stopping_on": "inner_validation" if verified else "unverified_or_legacy",
                          "training_group_ids": cfg.get("training_group_ids", []),
                          "inner_validation_group_ids": cfg.get("inner_validation_group_ids", []),
                          "held_out_group_ids": cfg.get("held_out_group_ids", []),
                          "probability_sha256": sha256(paths["probability_path"]),
                          "reference_labels_sha256": sha256(paths["reference_labels_path"]),
                          "prediction_labels_sha256": sha256(paths["prediction_labels_path"]),
                          "roi_sha256": sha256(paths["roi_path"]),
                          "calibration_um_per_pixel": float(cal[sid]["um_per_pixel"]),
                          "calibration_source_sha256": sha256(calibration),
                          "original_image_sha256": sha256(resolve_path(r["grayscale_path"], workspace)),
                          "original_boundary_sha256": sha256(resolve_path(r["training_mask_path"], workspace)),
                          "source_manifest_sha256": sha256(manifest), "source_fold_manifest_sha256": sha256(folds),
                          "prediction_postprocessing": config["prediction_postprocessing"],
                          "reference_postprocessing": config["reference_postprocessing"],
                          "inference": {"patch_size": patch_size, "stride": stride, "batch_size": batch_size,
                                        "overlap_blend": "Hann_window_floor_0.05", "device": str(device)}}
            write_json(prov_path, provenance)
            entries = {"sample_id": sid, "group_id": r["group_id"], "fold": fold_index[sid]["fold"],
                       "model_outer_fold": fold, "magnification_x": r.get("magnification_x", ""),
                       "um_per_pixel": cal[sid]["um_per_pixel"], "model_id": model_id, "run_id": run_id,
                       "evaluation_role": role, "provenance_status": "checkpoint_verified" if verified else "unverified",
                       "roi_origin": r.get("roi_origin") or "annotation_defined_legacy",
                       **{k: os.path.relpath(p, output) for k,p in paths.items()},
                       "provenance_path": os.path.relpath(prov_path, output)}
            manifests[role].append(entries)
            print(f"cache fold={fold} role={role} {sid}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for role, entries in manifests.items():
        if entries:
            write_csv(output / f"{role}_manifest.csv", entries)
    report = {"held_out_fields": len(manifests["held_out"]), "inner_fields": len(manifests["inner_validation"]),
              "model_id": model_id, "run_id": run_id, "verified_checkpoints": all(c[2] for c in checkpoints.values())}
    write_json(output / "cache_summary.json", report)
    return report
