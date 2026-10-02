"""Generate a small SYNTHETIC example with no private/material data or trained model."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from .io import write_csv, write_json, fresh_output


def make_demo(output: Path, seed: int = 20260928) -> Path:
    fresh_output(output)
    rng = np.random.default_rng(seed)
    rows = []
    # Six independent synthetic IDs; three have two fields (9 fields total).
    for group in range(6):
        for field in range(2 if group < 3 else 1):
            sid = f"SYNTHETIC_g{group}_f{field}"
            folder = output / sid
            folder.mkdir()
            shape = (112, 144)
            seeds = np.zeros(shape, bool)
            for y in range(7, shape[0], 18):
                for x in range(7, shape[1], 18):
                    yy = int(np.clip(y + rng.integers(-3, 4), 0, shape[0]-1))
                    xx = int(np.clip(x + rng.integers(-3, 4), 0, shape[1]-1))
                    seeds[yy, xx] = True
            seed_labels, _ = ndi.label(seeds)
            reference = watershed(np.zeros(shape), seed_labels, compactness=0.0, watershed_line=True).astype(np.int32)
            boundaries = reference == 0
            roi = np.ones(shape, bool)
            roi[-8:, -20:] = False
            reference[~roi] = 0
            # Deliberately perturbed signal; not the output of a neural network.
            amplitude = 0.70 + 0.04 * group
            probability = np.clip(ndi.gaussian_filter(ndi.binary_dilation(boundaries).astype(float), 0.45) * amplitude
                                  + rng.normal(0, 0.035, shape) + 0.03, 0, 1).astype(np.float32)
            # A few deterministic broken boundary patches ensure pipeline stress.
            if group % 2 == 0:
                probability[32:43, 45:74] *= 0.15
            np.save(folder / "reference_labels.npy", reference, allow_pickle=False)
            np.save(folder / "roi.npy", roi, allow_pickle=False)
            np.save(folder / "probability.npy", probability, allow_pickle=False)
            rows.append({"sample_id": sid, "group_id": f"SYNTHETIC_specimen_{group}", "fold": group % 3,
                         "magnification_x": 200 if group % 2 == 0 else 500, "um_per_pixel": 0.25 if group % 2 == 0 else 0.13,
                         "model_id": "SYNTHETIC_signal_not_neural_model", "run_id": f"demo_seed_{seed}",
                         "evaluation_role": "synthetic_demo", "provenance_status": "synthetic_demo", "roi_origin": "synthetic_fixed_roi",
                         "roi_path": f"{sid}/roi.npy", "reference_labels_path": f"{sid}/reference_labels.npy",
                         "probability_path": f"{sid}/probability.npy"})
    write_csv(output / "manifest.csv", rows)
    write_json(output / "NOTICE.json", {"synthetic_only": True, "seed": seed,
                                         "warning": "Functional fixture only. No empirical claim about steel, PAGB, or the manuscript."})
    return output / "manifest.csv"
