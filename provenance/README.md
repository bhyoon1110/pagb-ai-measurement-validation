# Public provenance record

This directory contains the non-identifying part of the verified run provenance.

- `fold_summary.csv`: outer fold, inner-validation fold, group counts, patch counts, best/stop epochs, selected threshold, and checkpoint SHA-256.
- `selected_parameters.json`: fold-specific inner-only postprocessing selections and manifest hashes with specimen identifiers removed.
- `training_logs/`: epoch-level training and validation metrics for each outer-fold model.
- `environment/`: Python and source-snapshot identifiers.

The original provenance archive contained 201 hashed records, field-level manifests, two cache roles (`inner_validation` and `held_out`), five model checkpoints, and the complete source snapshot. Field identifiers, cached predictions, raw labels, historical field-specific scripts, and checkpoint binaries are not included in public Git history. The public `pagb` package is the audited evaluation toolkit; its archived source identifier is retained under `environment/`. Checkpoint hashes remain public so an authorized copy can be verified byte-for-byte.

Postprocessing selection used the same inner fold used for checkpoint selection. It is not described as a second independent validation set. The outer held-out fields were not used for selection.
