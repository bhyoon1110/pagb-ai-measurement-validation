"""Frozen reference and configurable predicted postprocessing, CPU only."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import inspect
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, relabel_sequential
from .legacy_topology import _clean_topology


@dataclass(frozen=True)
class Postprocess:
    boundary_threshold: float = 0.55
    min_distance: int = 8
    min_grain_area: int = 40
    gaussian_sigma: float = 1.0
    topology_mode: str = "legacy"  # Preserve the paper's segmentation; audit separately.

    def validate(self):
        if not 0 < self.boundary_threshold < 1:
            raise ValueError("boundary_threshold must be in (0,1).")
        if (not isinstance(self.min_distance, (int, np.integer)) or isinstance(self.min_distance, bool)
                or not isinstance(self.min_grain_area, (int, np.integer)) or isinstance(self.min_grain_area, bool)):
            raise ValueError("min_distance and min_grain_area must be integers.")
        if self.min_distance < 1 or self.min_grain_area < 1 or self.gaussian_sigma < 0:
            raise ValueError("Invalid postprocessing size/sigma.")
        if not np.isfinite(self.gaussian_sigma):
            raise ValueError("gaussian_sigma must be finite.")
        if self.topology_mode not in ("legacy", "none"):
            raise ValueError("topology_mode must be legacy or none.")
        return self

    def to_dict(self):
        return asdict(self)


def remove_objects_below(mask: np.ndarray, minimum: int) -> np.ndarray:
    if "max_size" in inspect.signature(morphology.remove_small_objects).parameters:
        return morphology.remove_small_objects(mask, max_size=minimum - 1)
    return morphology.remove_small_objects(mask, min_size=minimum)


def fill_holes_below(mask: np.ndarray, minimum: int) -> np.ndarray:
    if "max_size" in inspect.signature(morphology.remove_small_holes).parameters:
        return morphology.remove_small_holes(mask, max_size=minimum - 1)
    return morphology.remove_small_holes(mask, area_threshold=minimum)


def watershed_labels(probability: np.ndarray, roi: np.ndarray, params: Postprocess,
                     return_audit: bool = False):
    params.validate()
    p = np.asarray(probability, np.float32)
    roi = np.asarray(roi, bool)
    if p.ndim != 2 or p.shape != roi.shape:
        raise ValueError("Probability and ROI must be matching 2D arrays.")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probability must be finite in [0,1].")
    if not roi.any():
        labels = np.zeros(p.shape, np.int32)
        audit = {"seed_count": 0, "removed_small_labels": 0, "topology_changed_px": 0,
                 "labels_before_min_area": 0, "labels_after_min_area": 0, "labels_after_topology": 0,
                 "topology_mode": params.topology_mode}
        return (labels, audit) if return_audit else labels
    smooth = cv2.GaussianBlur(p, (0, 0), params.gaussian_sigma) if params.gaussian_sigma else p.copy()
    space = roi & (smooth < params.boundary_threshold)
    size = max(8, params.min_grain_area // 8)
    space = remove_objects_below(space, size)
    space = fill_holes_below(space, size) & roi
    distance = ndi.distance_transform_edt(space)
    coords = peak_local_max(distance, labels=space, min_distance=params.min_distance,
                            threshold_abs=2, exclude_border=False)
    markers = np.zeros(p.shape, np.int32)
    if len(coords):
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    else:
        markers, _ = ndi.label(space)
    labels = watershed(smooth, markers=markers, mask=roi, watershed_line=True).astype(np.int32)
    counts = np.bincount(labels.ravel())
    small = np.flatnonzero((counts < params.min_grain_area) & (np.arange(counts.size) > 0))
    labels[np.isin(labels, small)] = 0
    before = labels.copy()
    if params.topology_mode == "legacy":
        labels = _clean_topology(labels, roi)
    changed = int(np.count_nonzero(before != labels))
    labels[~roi] = 0
    labels, _, _ = relabel_sequential(labels)
    audit = {"seed_count": int(markers.max(initial=0)), "removed_small_labels": int(len(small)),
             "labels_before_min_area": int(np.count_nonzero(counts[1:])),
             "labels_after_min_area": int(len(np.unique(before[before > 0]))),
             "labels_after_topology": int(labels.max(initial=0)),
             "topology_changed_px": changed, "topology_mode": params.topology_mode}
    return (labels.astype(np.int32), audit) if return_audit else labels.astype(np.int32)
