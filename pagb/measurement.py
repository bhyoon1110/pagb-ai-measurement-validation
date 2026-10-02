"""Versioned planimetry and explicit counterfactuals, not an ASTM certification.

legacy_v1 reproduces the historical local area rasterization (on ROI-clipped labels).
strict_v2 requires >=2 DISTINCT surviving neighbors for an included boundary pixel.
Both use the original E1382-table coefficients, avoiding a silent change in G scale.
The strict rule is a conservative discrete convention for 1-pixel watershed lines;
it is NOT an exact area rule for arbitrary thick voids. Compare both on real data.
"""
from __future__ import annotations
import math
import hashlib
import numpy as np
from scipy import ndimage as ndi

G_COEFFICIENT = 3.321928
G_OFFSET = 2.954
PROTOCOLS = ("legacy_v1", "strict_v2")


def validate_labels(labels: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Instance labels must be a 2D integer array, not probabilities/RGB.")
    if np.any(labels < 0):
        raise ValueError("Instance IDs must be nonnegative; zero is background/boundary.")
    if labels.size and np.max(labels) > np.iinfo(np.int64).max:
        raise ValueError("Instance ID exceeds int64 range.")
    if roi is not None and np.asarray(roi).shape != labels.shape:
        raise ValueError(f"Label/ROI shape mismatch: {labels.shape} vs {np.shape(roi)}")
    return labels.astype(np.int64, copy=False)


def compact_labels(labels: np.ndarray, roi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = validate_labels(labels, roi).copy()
    if roi is not None:
        arr[~np.asarray(roi, bool)] = 0
    ids = np.unique(arr[arr > 0])
    compact = np.zeros(arr.shape, np.int32)
    mask = arr > 0
    compact[mask] = np.searchsorted(ids, arr[mask]) + 1
    return compact, ids


def g_from_count_area(count: float, area_mm2: float | None) -> float | None:
    if area_mm2 is None or not (math.isfinite(area_mm2) and area_mm2 > 0):
        return None
    if not (math.isfinite(count) and count > 0):
        return None
    return G_COEFFICIENT * math.log10(count / area_mm2) - G_OFFSET


def border_ids(labels: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    roi = np.asarray(roi, bool)
    arr = validate_labels(labels, roi)
    border = roi & ~ndi.binary_erosion(roi, structure=np.ones((3, 3), bool), border_value=0)
    touching = np.unique(arr[border])
    touching = touching[touching > 0]
    present = np.unique(arr[roi & (arr > 0)])
    return np.setdiff1d(present, touching), touching


def measurement_masks(labels: np.ndarray, roi: np.ndarray, protocol: str = "strict_v2") -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown measurement protocol: {protocol}")
    roi = np.asarray(roi, bool)
    arr, _ = compact_labels(labels, roi)
    kept, deleted = border_ids(arr, roi)
    interiors = roi & np.isin(arr, kept)
    removed = roi & np.isin(arr, deleted)
    footprint = np.ones((3, 3), bool)
    # Historical rule only required a surviving neighbor, not two distinct grains.
    candidate = roi & (arr == 0) & ~ndi.binary_dilation(removed, footprint)
    if protocol == "legacy_v1":
        between = candidate & ndi.binary_dilation(interiors, footprint)
    else:
        kept_labels = np.where(interiors, arr, 0)
        sentinel = int(arr.max(initial=0)) + 1
        lo = ndi.minimum_filter(np.where(kept_labels > 0, kept_labels, sentinel),
                                size=3, mode="constant", cval=sentinel)
        hi = ndi.maximum_filter(kept_labels, size=3, mode="constant", cval=0)
        between = candidate & (lo < hi) & (lo != sentinel)
        # A boundary on the ROI edge is not an internal boundary.
        between &= ndi.binary_erosion(roi, footprint, border_value=0)
    return {"labels": arr, "kept_ids": kept, "deleted_ids": deleted,
            "interiors": interiors, "between": between, "area": interiors | between}


def measure(labels: np.ndarray, roi: np.ndarray, um_per_pixel: float | None,
            protocol: str = "strict_v2") -> dict:
    roi = np.asarray(roi, bool)
    original = validate_labels(labels, roi)
    if um_per_pixel is not None:
        um_per_pixel = float(um_per_pixel)
        if not math.isfinite(um_per_pixel) or um_per_pixel <= 0:
            raise ValueError("um_per_pixel must be finite and >0, or None when unavailable.")
    maps = measurement_masks(original, roi, protocol)
    n, deleted = len(maps["kept_ids"]), len(maps["deleted_ids"])
    area_px = int(maps["area"].sum())
    roi_px, frame_px = int(roi.sum()), int(roi.size)
    pixel_mm2 = (um_per_pixel / 1000.0) ** 2 if um_per_pixel is not None else None
    area_mm2 = area_px * pixel_mm2 if pixel_mm2 is not None else None
    roi_mm2 = roi_px * pixel_mm2 if pixel_mm2 is not None else None
    if not roi_px:
        status = "empty_roi"
    elif n == 0:
        status = "no_surviving_grains"
    elif area_px == 0:
        status = "zero_measured_area"
    elif pixel_mm2 is None:
        status = "missing_scale"
    else:
        status = "ok"
    g = g_from_count_area(n, area_mm2)
    result = {
        "measurement_protocol": protocol, "g_status": status, "g_defined": g is not None,
        "um_per_pixel": um_per_pixel,
        "roi_shape": tuple(roi.shape),
        "roi_mask_sha256": hashlib.sha256(str(roi.shape).encode() + np.packbits(roi).tobytes()).hexdigest(),
        "counted_grains": int(n), "deleted_grains": int(deleted),
        "present_grains": int(n + deleted), "planimetric_count": int(n),
        "measurement_px": area_px, "grain_interior_px": int(maps["interiors"].sum()),
        "included_boundary_px": int(maps["between"].sum()),
        "roi_px": roi_px, "frame_px": frame_px, "area_mm2": area_mm2,
        "roi_area_mm2": roi_mm2,
        "na_per_mm2": n / area_mm2 if g is not None else None,
        "astm_g": g, "d_plan_um": math.sqrt(area_mm2 / n) * 1000 if g is not None else None,
        "measurement_area_fraction_roi": area_px / roi_px if roi_px else None,
        "measurement_area_fraction_frame": area_px / frame_px if frame_px else None,
        "effective_roi_fraction": roi_px / frame_px if frame_px else None,
        "labels_outside_roi_px": int(np.count_nonzero((original > 0) & ~roi)),
        "jeffries_count": n + 0.5 * deleted,
        "jeffries_style_roi_g": g_from_count_area(n + 0.5 * deleted, roi_mm2),
        "numerator_only_half_count_g": g_from_count_area(n + 0.5 * deleted, area_mm2),
        "fixed_roi_count_only_g": g_from_count_area(n, roi_mm2),
        "count_ge_50_information_only": n >= 50,
    }
    return result


def counterfactual(pred: dict, ref: dict) -> dict:
    """Algebraic decomposition; fixed-reference-area is an oracle diagnostic, not accuracy.

    Shared-ROI comparison must put BOTH methods over that same ROI. It then has the
    same count-ratio delta as the fixed-reference-area comparison. Never compare a
    fixed-ROI prediction with a coupled reference and call that a pure count effect.
    """
    # Pixel-area decomposition assumes the same physical calibration. A changed
    # scale is a separate effect and must not be attributed to count/area coupling.
    ps, rs = pred.get("um_per_pixel"), ref.get("um_per_pixel")
    if ps is not None and rs is not None and not math.isclose(ps, rs, rel_tol=1e-12):
        raise ValueError("Counterfactual requires identical physical calibration on both sides.")
    def logratio(a, b):
        return math.log10(a / b) if a is not None and b is not None and a > 0 and b > 0 else None
    nlog = logratio(pred["counted_grains"], ref["counted_grains"])
    alog = logratio(pred["measurement_px"], ref["measurement_px"])
    ng = G_COEFFICIENT * nlog if nlog is not None else None
    ag = -G_COEFFICIENT * alog if alog is not None else None
    dg = ng + ag if ng is not None and ag is not None else None
    # Density-derived diameter ratio comes directly from A/N, not mean object ECD.
    size_ratio = 10 ** ((alog - nlog) / 2) if alog is not None and nlog is not None else None
    # Equal pixel COUNTS do not establish the same spatial region.
    same_roi = bool(pred.get("roi_mask_sha256") and
                    pred.get("roi_mask_sha256") == ref.get("roi_mask_sha256"))
    out = {
        "count_ratio": 10 ** nlog if nlog is not None else (0.0 if ref["counted_grains"] > 0 else None),
        "measurement_area_ratio": 10 ** alog if alog is not None else (0.0 if ref["measurement_px"] > 0 else None),
        "count_component_delta_g": ng, "area_component_delta_g": ag,
        "coupled_delta_g_dimensionless": dg,
        "fixed_reference_area_delta_g": ng,
        "common_roi_verified": same_roi,
        "absolute_g_attenuation": abs(ng) - abs(dg) if ng is not None and dg is not None else None,
        "attenuation_interpretation": "positive=smaller_disagreement;negative=amplification;NOT_accuracy_gain",
        "count_area_opposing_signs": ng * ag < 0 if ng is not None and ag is not None else None,
        "count_area_sign_reversal": ng * dg < 0 if ng is not None and dg is not None else None,
        "common_roi_both_sides_delta_g": ng if same_roi else None,
        "d_plan_ratio": size_ratio,
        "abs_size_deviation": abs(size_ratio - 1) if size_ratio is not None else None,
        "oracle_fixed_reference_area_pred_g": g_from_count_area(pred["counted_grains"], ref["area_mm2"]),
    }
    if pred["astm_g"] is not None and ref["astm_g"] is not None:
        out["delta_g"] = pred["astm_g"] - ref["astm_g"]
        out["abs_delta_g"] = abs(out["delta_g"])
        out["decomposition_residual_g"] = out["delta_g"] - dg if dg is not None else None
    else:
        out.update(delta_g=None, abs_delta_g=None, decomposition_residual_g=None)
    return out
