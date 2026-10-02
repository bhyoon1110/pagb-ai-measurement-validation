"""Read-only bridge from locked result files to manuscript tables and author queries.

No manuscript values are hardcoded, no missing numbers are imputed, and this tool
never replaces prose in a manuscript automatically. Verified metadata is evidence
of software consistency, NOT independent proof of specimen identity/accuracy.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
from .io import read_csv, write_csv, write_json, sha256, fresh_output, finite
from .statistics import summarize, failure_counts

IDENTIFIERS = {"sample_id", "group_id", "model_id", "run_id", "evaluation_role",
               "provenance_status", "roi_origin", "prediction_source", "calibration_provenance_status"}


def typed_rows(path: Path) -> list[dict]:
    rows = []
    for raw in read_csv(path):
        row = {}
        for key, value in raw.items():
            if key in IDENTIFIERS or key.endswith("sha256"):
                row[key] = value
            elif value == "":
                row[key] = None
            elif finite(value):
                row[key] = float(value)
            else:
                row[key] = value
        rows.append(row)
    return rows


def seal_result_tables(output: Path) -> None:
    names = [p.name for p in sorted(output.iterdir()) if p.suffix in (".json", ".csv")
             and p.name != "result_table_integrity.json"]
    write_json(output / "result_table_integrity.json", {
        "schema": "pagb_result_table_integrity_v1", "files": {name: sha256(output / name) for name in names},
        "note": "Hash consistency check, not cryptographic authorship or proof of scientific validity."})


def export_paper(run_dir: Path, output: Path, expected_fields: int | None = None,
                 expected_specimens: int | None = None, require_verified: bool = False) -> dict:
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "per_field_metrics.csv").is_file() and (run_dir / "evaluation").is_dir():
        run_dir = run_dir / "evaluation"
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if meta.get("command") != "evaluate":
        raise ValueError("paper-export needs an evaluate result, not a sweep or corrupted-label experiment.")
    seal_path = run_dir / "result_table_integrity.json"
    if not seal_path.is_file():
        raise ValueError("Missing v3 result integrity record. Rerun evaluation; do not relabel old output as v3.")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for name, expected in seal["files"].items():
        if Path(name).name != name or not (run_dir / name).is_file() or sha256(run_dir / name) != expected:
            raise ValueError(f"Result table changed after evaluation: {name}")
    rows = typed_rows(run_dir / "per_field_metrics.csv")
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate per-field results.")
    calculated = summarize(rows)
    stored = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for key, actual in calculated.items():
        original = stored.get(key)
        same = math.isclose(float(actual), float(original), rel_tol=1e-9, abs_tol=1e-10) if finite(actual) and finite(original) else actual == original
        if not same:
            raise ValueError(f"Summary/per-field mismatch for {key}; rerun evaluation.")
    if expected_fields is not None and len(rows) != expected_fields:
        raise ValueError(f"Expected {expected_fields} fields; found {len(rows)}.")
    if expected_specimens is not None and calculated["n_specimens"] != expected_specimens:
        raise ValueError(f"Expected {expected_specimens} specimens; found {calculated['n_specimens']}.")
    issues = []
    synthetic = any(r.get("provenance_status") == "synthetic_demo" for r in rows) or bool(meta.get("synthetic_only"))
    if synthetic:
        issues.append("SYNTHETIC DEMO: numbers are not manuscript experimental results.")
    if meta.get("provenance_statuses") != ["checkpoint_verified"] or meta.get("evaluation_roles") != ["held_out"]:
        issues.append("Checkpoint/split provenance is not verified held-out evaluation.")
    if any(r.get("calibration_provenance_status") != "recorded_and_checked" for r in rows):
        issues.append("Physical calibration was not bound to all prediction caches.")
    if any(r.get("roi_origin") in (None, "", "unspecified_must_be_reported") for r in rows):
        issues.append("ROI construction must be specified by the authors.")
    if any(not finite(r.get("um_per_pixel")) for r in rows):
        issues.append("Missing physical calibration; absolute G cannot be reported for every side.")
    selection = meta["config"].get("postprocessing_selection_status", "")
    if selection not in ("inner_only_same_outer_model", "author_locked_before_heldout_evaluation"):
        issues.append("Fixed postprocessing was not recorded as selected independently of held-out results.")
    if any(r.get("cached_postprocessing_matches_requested") == "False" for r in rows):
        issues.append("Explicit cached replay differs from requested postprocessing; not the fixed sweep baseline.")
    if require_verified and issues:
        raise ValueError("Paper-export verification blocked:\n- " + "\n- ".join(issues))
    fresh_output(output)
    ci = json.loads((run_dir / "bootstrap_ci.json").read_text(encoding="utf-8")).get("intervals", {})
    keys = ("g_mae", "g_rmse", "g_bias", "instance_f1_micro", "instance_f1_field_mean", "pooled_count_ratio",
            "pred_measured_area_fraction_roi_mean", "reference_measured_area_fraction_roi_mean",
            "pred_measured_area_fraction_frame_mean", "reference_measured_area_fraction_frame_mean",
            "geometric_mean_size_ratio", "median_absolute_size_deviation", "area_count_slope", "prediction_undefined_rate")
    table = []
    for key in keys:
        interval = ci.get(key, {})
        table.append({"metric": key, "estimate": calculated[key], "lower_95": interval.get("lower_95"),
                      "upper_95": interval.get("upper_95"), "defined_bootstrap_replicates": interval.get("defined_replicates"),
                      "n_all_fields": len(rows), "n_G_pairs": calculated["both_defined"],
                      "source": "summary.json + bootstrap_ci.json", "synthetic_only": synthetic})
    write_csv(output / "paper_main_table.csv", table)
    write_json(output / "paper_failure_counts.json", failure_counts(rows))
    report = {"source_run": str(run_dir), "source_integrity_sha256": sha256(seal_path),
              "synthetic_only": synthetic, "status": "blocked_pending_author_checks" if issues else "software_checks_passed_author_review_required",
              "submission_ready": False, "issues": issues,
              "notice": "No automatic submission certification. Confirm physical specimens/ROI, final reference and rasterization, "
                        "fair model training, data permissions, and all author queries in the revised manuscript.",
              "n_fields": len(rows), "n_specimens": calculated["n_specimens"]}
    write_json(output / "paper_export_status.json", report)
    write_json(output / "paper_verified_values.json", {"status": report["status"], "synthetic_only": synthetic,
               "statistics": calculated, "failure_counts": failure_counts(rows)})
    f = failure_counts(rows)
    title = "SYNTHETIC DEMO — NOT MANUSCRIPT RESULTS" if synthetic else "Run-derived manuscript numbers — author verification required"
    lines = ["# " + title, "", report["notice"], "", "## Field denominators", "",
             f"Total {f['n_total']}; both G defined {f['both_defined']}; reference-only undefined {f['reference_only_undefined']}; "
             f"prediction-only undefined {f['prediction_only_undefined']}; both undefined {f['both_undefined']}.",
             f"Prediction failures {f['prediction_undefined_total']}; reference failures {f['reference_undefined_total']}; "
             f"union excluded {f['excluded_from_paired_g']}.", "", "## Blocking checks / scope", ""]
    lines += ["- " + s for s in issues] or ["Software checks passed; substantive author verification remains."]
    lines += ["", "## Manuscript mapping", "", "Section 3.1: paper_main_table.csv and paper_failure_counts.json.",
              "Section 2.7 / additional validation: coupling_diagnostic_summary.json in the source evaluation.",
              "Section 2.8: instance_matches_iou05.csv and boundary F1 columns in per_field_metrics.csv.",
              "Section 2.9: sweep_summary.csv, common_cohort.json, paired_setting_bootstrap_ci.csv.",
              "Section 2.10: corruption_per_field.csv; repetitions do not constitute additional specimens.",
              "Never fill a missing result with an old manuscript value or a synthetic value."]
    (output / "PAPER_NUMBERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
