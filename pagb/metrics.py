"""Object matching on the complete common ROI, before measurement-driven deletion.

A reference object and a predicted object can each be used at most once.
Primary objective: maximize number of pairs >= IoU threshold; secondary: total IoU.
Empty/empty is undefined, not perfect. Counts and denominators always accompany F1.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries
from .measurement import compact_labels, validate_labels


def match_instances(reference: np.ndarray, prediction: np.ndarray, roi: np.ndarray,
                    iou_threshold: float = 0.5, max_matrix_cells: int = 25_000_000) -> dict:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1].")
    roi = np.asarray(roi, bool)
    ref, ref_ids = compact_labels(reference, roi)
    pred, pred_ids = compact_labels(prediction, roi)
    nr, npred = len(ref_ids), len(pred_ids)
    if (nr + 1) * (npred + 1) > max_matrix_cells:
        raise ValueError("Too many instance pairs for dense matching; inspect over-segmentation or raise max_matrix_cells explicitly.")
    matches = []
    merge_candidates = split_candidates = 0
    if nr and npred:
        flat = ref[roi].astype(np.int64) * (npred + 1) + pred[roi]
        contingency = np.bincount(flat, minlength=(nr + 1) * (npred + 1)).reshape(nr + 1, npred + 1)
        ref_area = contingency.sum(axis=1)[1:]
        pred_area = contingency.sum(axis=0)[1:]
        overlap = contingency[1:, 1:]
        union = ref_area[:, None] + pred_area[None, :] - overlap
        iou = overlap / np.maximum(union, 1)
        eligible = (iou >= iou_threshold) & (overlap > 0)
        # One extra admissible match outweighs every possible IoU tie-break gain.
        score = eligible * (min(nr, npred) + 1.0) + np.where(eligible, iou, 0.0)
        rows, cols = linear_sum_assignment(score, maximize=True)
        for r, c in zip(rows, cols):
            if eligible[r, c]:
                matches.append({"reference_id": int(ref_ids[r]), "prediction_id": int(pred_ids[c]),
                                "iou": float(iou[r, c])})
        # Descriptive overlap flags, not a complete taxonomic error classifier.
        merge_candidates = int(((overlap / np.maximum(ref_area[:, None], 1) >= 0.2).sum(axis=0) >= 2).sum())
        split_candidates = int(((overlap / np.maximum(pred_area[None, :], 1) >= 0.2).sum(axis=1) >= 2).sum())
    tp = len(matches)
    fp, fn = npred - tp, nr - tp
    precision = tp / npred if npred else (0.0 if nr else None)
    recall = tp / nr if nr else None
    f1 = 2 * tp / (nr + npred) if nr + npred else None
    return {"iou_threshold": iou_threshold, "n_reference": nr, "n_prediction": npred,
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "matched_mean_iou": float(np.mean([m["iou"] for m in matches])) if matches else None,
            "merge_overlap_candidates": merge_candidates, "split_overlap_candidates": split_candidates,
            "matches": matches}


def boundary_map(labels: np.ndarray, roi: np.ndarray) -> np.ndarray:
    arr = validate_labels(labels, roi).copy()
    roi = np.asarray(roi, bool)
    arr[~roi] = 0
    # Remove crop/artificial ROI edge from BOTH boundaries, independent of prediction.
    interior = ndi.binary_erosion(roi, structure=np.ones((3, 3), bool), border_value=0)
    return find_boundaries(arr, connectivity=2, mode="inner") & interior


def boundary_f1(reference: np.ndarray, prediction: np.ndarray, roi: np.ndarray,
                tolerance_px: float) -> dict:
    if not np.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("Boundary tolerance must be finite and >=0.")
    ref = np.asarray(reference, bool) & np.asarray(roi, bool)
    pred = np.asarray(prediction, bool) & np.asarray(roi, bool)
    if ref.shape != pred.shape:
        raise ValueError("Boundary map shapes differ.")
    nr, npred = int(ref.sum()), int(pred.sum())
    if not nr and not npred:
        return {"precision": None, "recall": None, "f1": None, "n_reference": 0, "n_prediction": 0}
    if not nr or not npred:
        return {"precision": 0.0, "recall": 0.0 if nr else None, "f1": 0.0,
                "n_reference": nr, "n_prediction": npred}
    # Pixel-neighborhood correspondence, not one-to-one object matching.
    matched_p = int(np.count_nonzero(pred & (ndi.distance_transform_edt(~ref) <= tolerance_px)))
    matched_r = int(np.count_nonzero(ref & (ndi.distance_transform_edt(~pred) <= tolerance_px)))
    precision, recall = matched_p / npred, matched_r / nr
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "n_reference": nr, "n_prediction": npred}
