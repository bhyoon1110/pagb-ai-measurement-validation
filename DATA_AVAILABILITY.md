# Data availability

## Public in this repository

- evaluation and analysis source code;
- aggregate numerical results and confidence intervals;
- manuscript figures generated from the verified run;
- fold-level training logs, counts, selected thresholds, and checkpoint SHA-256 hashes;
- environment and source-snapshot identifiers.

## Not included in public Git history

- raw optical micrographs and EBSD maps;
- manual annotation images and instance-label arrays;
- cached probability maps and intervention arrays;
- specimen-level manifests containing institutional identifiers;
- U-Net checkpoint binaries;
- the full field-level provenance archive.

These materials are excluded because their release requires confirmation of institutional data permissions and, for the large binary artifacts, a more appropriate archival channel. Subject to those permissions, requests may be directed to the corresponding author listed in the manuscript.

The public aggregate outputs are sufficient to audit the manuscript's reported denominators and scalar summaries, but not to retrain the model or independently reconstruct every per-field measurement from raw data.
