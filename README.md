# PAGB AI measurement validation

Provenance-aware evaluation code and aggregate results for automated prior-austenite grain-boundary (PAGB) segmentation and ASTM E1382-style planimetric grain-size measurement.

This repository accompanies the manuscript **"Validating deep learning for prior austenite grain sizing: Measurement-area coupling can mask segmentation errors."** It focuses on the distinction between segmentation agreement and the downstream scalar grain-size measurement.

## What is included

- `pagb/`: audited evaluation, sensitivity, corruption, provenance, and paper-export toolkit.
- `configs/`: synthetic smoke-test and verified analysis configurations.
- `provenance/`: public fold-level training summaries, training logs, checkpoint hashes, and inner-only postprocessing selections.
- `results/`: aggregate evaluation, sensitivity, intervention, confidence-interval, and figure outputs used for the manuscript.
- `tests/`: a synthetic smoke test that does not use research data or trained models.

## Headline results

| Quantity | Verified result |
|---|---:|
| Evaluation fields | 82 fields from 69 specimens |
| Paired fields with both G values defined | 78 |
| G mean absolute error | 0.510 (specimen-cluster 95% CI 0.355-0.704) |
| Micro-averaged instance F1 at IoU >= 0.5 | 0.544 (0.516-0.570) |
| Retained-count ratio, prediction/reference | 0.656 |
| Mean ROI-normalized measured-area fraction, prediction/reference | 0.327 / 0.520 |
| Fields in which the area term attenuated count-only G discrepancy | 58 of 78 |

The scalar G result and the object-level correspondence result answer different questions. The reported count ratio is not recall, and the counterfactual fixed-area calculation is a diagnostic rather than an alternative ASTM measurement.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -e .
python -m pagb demo --output demo_data
python -m pagb evaluate \
  --manifest demo_data/manifest.csv \
  --output demo_run \
  --config configs/smoke.json \
  --prediction-source probability \
  --allow-unverified-provenance
```

The demo is explicitly synthetic. It verifies that the public code executes; it does not reproduce any empirical claim about steel or PAGB segmentation.

## Verified workflow

The manuscript run used specimen-wise outer folds, checkpoint selection and postprocessing selection on the corresponding inner fold, fixed cached predictions for the held-out evaluation, field-specific calibration records, and specimen-cluster bootstrap intervals.

Fold-level settings and checkpoint hashes are in [`provenance/fold_summary.csv`](provenance/fold_summary.csv). The selected outer-fold thresholds were 0.35, 0.40, 0.35, 0.50, and 0.40 for folds 0-4. The common threshold sweep is descriptive and was not used to choose a new held-out-test threshold.

## Reproducing the manuscript analysis

The public repository contains code and aggregate outputs, but not the raw micrographs, annotations, cached probability maps, or trained weights. With institution-authorized inputs arranged according to the manifest schema, the verified stages are:

```bash
python -m pagb tune \
  --inner-manifest path/to/inner_validation_manifest.csv \
  --heldout-manifest path/to/held_out_manifest.csv \
  --output selected_parameters \
  --config configs/paper_verified.json

python -m pagb all \
  --manifest path/to/held_out_manifest.csv \
  --selected-parameters selected_parameters/selected_parameters.json \
  --output verified_run \
  --config configs/paper_verified.json \
  --prediction-source probability

python -m pagb paper-export \
  --run-dir verified_run \
  --output verified_run/paper \
  --expected-fields 82 \
  --expected-specimens 69 \
  --require-verified
```

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for the boundary between public artifacts and restricted research data.

## Verification status

The included run passed the software and integrity checks recorded in `results/`. The machine-generated paper-export status intentionally remains `software_checks_passed_author_review_required`: physical interpretation, specimen/ROI identity, data permissions, and final manuscript statements require author confirmation and cannot be certified automatically by software.

## Citation

Use [`CITATION.cff`](CITATION.cff), or cite the associated manuscript and this repository URL:

`https://github.com/bhyoon1110/pagb-ai-measurement-validation`

The manuscript-linked snapshot is archived as [release v1.0.0](https://github.com/bhyoon1110/pagb-ai-measurement-validation/releases/tag/v1.0.0).

## License

The source code is released under the [MIT License](LICENSE). Aggregate result tables, non-identifying provenance summaries, and figures under `results/` and `provenance/` are released under [CC BY 4.0](LICENSE-DATA). Restricted research data are not distributed and are outside these licenses.
