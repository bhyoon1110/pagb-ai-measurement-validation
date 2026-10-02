"""Historical uncapped topology rule, retained explicitly for replay/ablation.
Not evidence that enclosed grains cannot exist. Do not use as an unreported cleanup.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

def _clean_topology(labels: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Historical uncapped enclosed-label merging and single-neighbor hole filling.

    Kept for a controlled legacy/none comparison, not as proof that enclosed
    objects are artifacts or that this operation always preserves real grains.
    """
    neighbourhood = np.ones((3, 3), bool)

    def sole_neighbour(source: np.ndarray, value: int, box, exclude: int) -> tuple[int, np.ndarray, tuple]:
        pad = tuple(slice(max(s.start - 4, 0), s.stop + 4) for s in box)
        mask = source[pad] == value
        # 3 iterations cross the 1 px watershed line with margin for diagonals.
        ring = ndi.binary_dilation(mask, neighbourhood, iterations=3) & ~mask
        adjacent = np.unique(labels[pad][ring])
        adjacent = adjacent[(adjacent > 0) & (adjacent != exclude)]
        host = int(adjacent[0]) if len(adjacent) == 1 else 0
        return host, mask, pad

    # Only a grain that is fully enclosed can be an island. Anything touching the
    # frame border has a legitimate reason to have one neighbour (a corner grain),
    # and merging it would corrupt the host's area.
    frame_edge = roi & ~ndi.binary_erosion(roi, neighbourhood)
    on_border = set(np.unique(labels[frame_edge]).tolist())
    for value, box in enumerate(ndi.find_objects(labels), 1):
        if box is None or value in on_border:
            continue
        host, mask, pad = sole_neighbour(labels, value, box, exclude=value)
        if host:
            labels[pad][mask] = host

    holes, count = ndi.label(roi & (labels == 0))
    for value, box in enumerate(ndi.find_objects(holes), 1):
        if box is None:
            continue
        host, mask, pad = sole_neighbour(holes, value, box, exclude=0)
        if host:
            labels[pad][mask] = host
    return labels
