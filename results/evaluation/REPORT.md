# PAGB evaluation report

These numbers refer to the specified annotation and ROI, not an independently established physical ground truth.

## Denominators
Fields: 82; specimens: 69; paired G fields: 78.
Prediction undefined: 1; reference undefined: 3; union excluded: 4.

## Main results
G MAE: 0.509502; G bias: 0.202916.
Object F1 (micro): 0.543831; mean per-field F1: 0.49064.
Pooled count ratio (NOT recall): 0.656076.
Geometric mean density-derived size ratio: 0.932091; median absolute relative deviation: 0.0935383.
Area/count log slope: 0.969597; OLS SE (NOT cluster CI): 0.155617.

## Interpretation and provenance
Use bootstrap_ci.json for specimen-cluster intervals and defined replicate counts. CIs condition on cached predictions; they do not measure training-seed variability.
Check measurement_protocol_audit.csv before replacing legacy manuscript values. strict_v2 changes the rasterized area convention.
The numerator-only half-count variant is a hybrid diagnostic, NOT a complete Jeffries calculation.
Diagnostic G/F1 cutoffs are exploratory, not industrial pass/fail limits.
Annotation-defined ROIs must be reported; these evaluations do not establish fully automatic ROI localization.
See run_metadata.json and input_hashes.csv for software/configuration and exact inputs.
