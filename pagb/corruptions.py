"""Controlled label interventions, explicitly NOT newly observed microstructures.

Selection erases eligible INTERIOR objects and leaves the remaining objects intact.
It is a sampling/selection control, not a physical model of broken PAG boundaries.
Merge/split/border-merge are distinct interventions. Report realized operations;
adjacency/size constraints can prevent reaching the requested fraction.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from .measurement import compact_labels, border_ids

KINDS = ("random_selection", "small_first_selection", "large_first_selection", "merge", "split", "merge_to_border")


def _neighbor_extrema(arr: np.ndarray):
    sentinel = int(arr.max(initial=0)) + 1
    lo = ndi.minimum_filter(np.where(arr > 0, arr, sentinel), size=3, mode="constant", cval=sentinel)
    hi = ndi.maximum_filter(arr, size=3, mode="constant", cval=0)
    return lo, hi


def adjacency_pairs(labels: np.ndarray, roi: np.ndarray) -> list[tuple[int, int]]:
    """8-neighbor contact or a one-pixel separator with exactly two neighbors.

    Vectorized over boundary pixels. No per-background-pixel Python loop.
    """
    arr = np.where(roi, labels, 0)
    chunks = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y1, y2 = (slice(0, -dy), slice(dy, None)) if dy else (slice(None), slice(None))
        if dx > 0:
            x1, x2 = slice(0, -dx), slice(dx, None)
        elif dx < 0:
            x1, x2 = slice(-dx, None), slice(0, dx)
        else:
            x1 = x2 = slice(None)
        a, b = arr[y1, x1], arr[y2, x2]
        valid = (a > 0) & (b > 0) & (a != b)
        if valid.any():
            chunks.append(np.sort(np.stack((a[valid], b[valid]), axis=1), axis=1))
    lo, hi = _neighbor_extrema(arr)
    yy, xx = np.where(roi & (arr == 0) & (lo < hi))
    if len(yy):
        a, b = lo[yy, xx], hi[yy, xx]
        padded = np.pad(arr, 1)
        third = np.zeros(len(yy), bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v = padded[yy + 1 + dy, xx + 1 + dx]
                third |= (v > 0) & (v != a) & (v != b)
        if (~third).any():
            chunks.append(np.stack((a[~third], b[~third]), axis=1))
    if not chunks:
        return []
    return [tuple(map(int, x)) for x in np.unique(np.concatenate(chunks), axis=0)]


def _merge_graph(labels: np.ndarray, roi: np.ndarray, requested: int,
                 border: np.ndarray, kind: str, rng: np.random.Generator):
    """Contract original adjacency edges; rasterize labels only once at the end.

    Graph adjacency stays rooted in the original labels. This avoids introducing
    spurious adjacency across newly filled junction pixels at each operation.
    """
    edges = adjacency_pairs(labels, roi)
    parent = np.arange(int(labels.max(initial=0)) + 1)
    is_border = np.isin(parent, border)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return int(a)
    operations = 0
    for _ in range(requested):
        eligible = set()
        for x, y in edges:
            a, b = find(x), find(y)
            if a == b:
                continue
            ok = (not is_border[a] and not is_border[b]) if kind == "merge" else (is_border[a] != is_border[b])
            if ok:
                eligible.add(tuple(sorted((a,b))))
        if not eligible:
            break
        pairs = sorted(eligible)
        a, b = pairs[int(rng.integers(len(pairs)))]
        parent[b] = a
        is_border[a] |= is_border[b]
        operations += 1
    mapping = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    merged = mapping[labels]
    # Fill a separator only when at least two ORIGINAL adjacent labels collapsed
    # to a single connected graph component. Never fill an unrelated isolated void.
    lo_old, hi_old = _neighbor_extrema(labels)
    lo_new, hi_new = _neighbor_extrema(merged)
    fill = roi & (labels == 0) & (lo_old < hi_old) & (lo_new == hi_new) & (hi_new > 0)
    merged[fill] = hi_new[fill]
    return merged, operations


def corrupt(reference: np.ndarray, roi: np.ndarray, kind: str, strength: float,
            rng: np.random.Generator, min_piece_area: int = 1) -> tuple[np.ndarray, dict]:
    if kind not in KINDS or not 0 <= strength <= 1:
        raise ValueError("Invalid corruption kind/strength.")
    labels, _ = compact_labels(reference, roi)
    kept, border = border_ids(labels, roi)
    n_start = len(kept)
    requested = min(n_start, int(round(strength * n_start)))
    operations = 0
    if kind.endswith("selection"):
        ids = kept.copy()
        if kind == "random_selection":
            rng.shuffle(ids)
        else:
            areas = np.bincount(labels.ravel())
            ids = np.array(sorted(ids, key=lambda k: (areas[k], int(k)), reverse=kind.startswith("large")))
        removed = ids[:requested]
        labels[np.isin(labels, removed)] = 0
        operations = len(removed)
    elif kind in ("merge", "merge_to_border"):
        if requested:
            labels, operations = _merge_graph(labels, roi, requested, border, kind, rng)
    elif kind == "split":
        candidates = kept.copy()
        rng.shuffle(candidates)
        next_id = int(labels.max(initial=0)) + 1
        for ident in candidates:
            if operations >= requested:
                break
            yy, xx = np.where(labels == ident)
            axis = yy if np.ptp(yy) >= np.ptp(xx) else xx
            cut = int(np.median(axis))
            left, right = axis < cut, axis > cut
            if min(left.sum(), right.sum()) < min_piece_area:
                continue
            labels[yy[axis == cut], xx[axis == cut]] = 0
            labels[yy[right], xx[right]] = next_id
            next_id += 1
            operations += 1
    labels, _ = compact_labels(labels, roi)
    meta = {"corruption_kind": kind, "requested_strength": strength,
            "eligible_interior_grains_at_start": n_start, "requested_operations": requested,
            "realized_operations": operations,
            "realized_operations_per_initial_interior_grain": operations / n_start if n_start else 0.0,
            "is_selection_control": kind.endswith("selection")}
    return labels, meta
