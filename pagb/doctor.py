"""Source archive inventory and input preflight; never claims absent files exist."""
from __future__ import annotations
import ast
from pathlib import Path
from .io import write_json, environment

PUBLIC_REQUIRED = {
    "aggregate_results": ["results/evaluation/summary.json", "results/paper/paper_verified_values.json"],
    "public_provenance": ["provenance/fold_summary.csv", "provenance/selected_parameters.json"],
    "verified_configuration": ["configs/paper_verified.json"],
}

RESTRICTED_REPRODUCTION_INPUTS = {
    "field_level_manifests": ["private/held_out_manifest.csv", "private/inner_validation_manifest.csv"],
    "raw_or_cached_inputs": ["private/raw_images", "private/cached_probability_maps"],
    "model_checkpoints": ["private/checkpoints"],
}


def doctor(workspace: Path, output: Path | None = None) -> dict:
    workspace = Path(workspace).resolve()
    checks = {
        stage: [{"path": p, "exists": (workspace / p).exists()} for p in paths]
        for stage, paths in {**PUBLIC_REQUIRED, **RESTRICTED_REPRODUCTION_INPUTS}.items()
    }
    syntax_errors = []
    source_files = list((workspace / "pagb").glob("*.py"))
    for p in source_files:
        try:
            ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_errors.append({"path": str(p), "error": str(exc)})
    report = {
        "workspace": str(workspace),
        "environment": environment(),
        "input_checks": checks,
        "python_files_checked": len(source_files),
        "syntax_errors": syntax_errors,
        "stage_inputs_present": {stage: all(x["exists"] for x in parts) for stage, parts in checks.items()},
        "public_artifacts_complete": all(
            all(item["exists"] for item in checks[stage]) for stage in PUBLIC_REQUIRED
        ),
        "note": (
            "Restricted raw data and checkpoints are intentionally absent from the public repository. "
            "Their absence does not invalidate the included aggregate result audit, but prevents full retraining or per-field replay."
        ),
    }
    if output:
        write_json(output, report)
    return report
