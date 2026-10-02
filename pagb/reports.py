"""Figures and a numerical summary generated ONLY from this run's exported tables."""
from __future__ import annotations
from pathlib import Path
import json
import textwrap
import numpy as np
from .io import read_csv, finite


def _number(value):
    return float(value) if finite(value) else None


def _fmt(value):
    return f"{value:.6g}" if finite(value) else "undefined / not estimated"


def report(run_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    synthetic = meta.get("synthetic_only", False)
    prefix = "SYNTHETIC DEMO — NOT MANUSCRIPT RESULTS — " if synthetic else ""
    output = run_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    produced = []

    def finish(fig, ax, stem, title):
        ax.set_title("\n".join(textwrap.wrap(prefix + title, width=56)), fontsize=10)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"{stem}.png", dpi=180)
        fig.savefig(output / f"{stem}.svg")
        plt.close(fig)
        produced.append(str(output / f"{stem}.png"))

    if (run_dir / "per_field_metrics.csv").is_file():
        raw = read_csv(run_dir / "per_field_metrics.csv")
        pairs = [r for r in raw if finite(r["abs_delta_g"]) and finite(r["instance_all_f1"])]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([float(r["abs_delta_g"]) for r in pairs], [float(r["instance_all_f1"]) for r in pairs], alpha=0.7)
        ax.set(xlabel="Absolute G disagreement with reference", ylabel="All-ROI instance F1 (IoU >= 0.5)", ylim=(-0.03, 1.03))
        finish(fig, ax, "F1_instance_vs_G", f"Instance agreement versus G: {len(pairs)}/{len(raw)} fields evaluable")
        valid = [r for r in raw if finite(r["count_ratio"]) and finite(r["measurement_area_ratio"])
                 and float(r["count_ratio"]) > 0 and float(r["measurement_area_ratio"]) > 0]
        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.array([float(r["count_ratio"]) for r in valid])
        y = np.array([float(r["measurement_area_ratio"]) for r in valid])
        ax.scatter(x, y, alpha=0.7)
        if len(x):
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], linestyle="--", label="Equal relative count/area change")
            ax.set(xscale="log", yscale="log")
            ax.legend()
        ax.set(xlabel="Count ratio (prediction / reference)", ylabel="Measured-area ratio")
        finish(fig, ax, "F2_count_area_coupling", f"Count–area decomposition, n={len(valid)}")
        fig, ax = plt.subplots(figsize=(6, 5))
        valid = [r for r in raw if finite(r["fixed_reference_area_delta_g"]) and finite(r["coupled_delta_g_dimensionless"])]
        x = np.array([abs(float(r["fixed_reference_area_delta_g"])) for r in valid])
        y = np.array([abs(float(r["coupled_delta_g_dimensionless"])) for r in valid])
        ax.scatter(x, y, alpha=0.7)
        if len(x):
            lim = max(x.max(), y.max(), 0.1)
            ax.plot([0, lim], [0, lim], linestyle="--")
        ax.set(xlabel="Absolute count-only delta G (diagnostic)", ylabel="Absolute coupled count/area delta G")
        finish(fig, ax, "F3_counterfactual", "Algebraic diagnostic — fixed area is not an accuracy reference")
        summary = json.loads((run_dir / "summary.json").read_text())
        note = ["# " + prefix + "PAGB evaluation report", "",
                "This report is generated from this run only. Synthetic values must never be used as manuscript results." if synthetic else
                "These numbers refer to the specified annotation and ROI, not an independently established physical ground truth.", "",
                "## Denominators", f"Fields: {summary['n_total']}; specimens: {summary['n_specimens']}; paired G fields: {summary['both_defined']}.",
                f"Prediction undefined: {summary['prediction_undefined_total']}; reference undefined: {summary['reference_undefined_total']}; union excluded: {summary['excluded_from_paired_g']}.", "",
                "## Main results", f"G MAE: {_fmt(summary['g_mae'])}; G bias: {_fmt(summary['g_bias'])}.",
                f"Object F1 (micro): {_fmt(summary['instance_f1_micro'])}; mean per-field F1: {_fmt(summary['instance_f1_field_mean'])}.",
                f"Pooled count ratio (NOT recall): {_fmt(summary['pooled_count_ratio'])}.",
                f"Geometric mean density-derived size ratio: {_fmt(summary['geometric_mean_size_ratio'])}; median absolute relative deviation: {_fmt(summary['median_absolute_size_deviation'])}.",
                f"Area/count log slope: {_fmt(summary['area_count_slope'])}; OLS SE (NOT cluster CI): {_fmt(summary['area_count_slope_ols_se_not_cluster_ci'])}.", "",
                "## Interpretation and provenance",
                "Use bootstrap_ci.json for specimen-cluster intervals and defined replicate counts. CIs condition on cached predictions; they do not measure training-seed variability.",
                "Check measurement_protocol_audit.csv before replacing legacy manuscript values. strict_v2 changes the rasterized area convention.",
                "The numerator-only half-count variant is a hybrid diagnostic, NOT a complete Jeffries calculation.",
                "Diagnostic G/F1 cutoffs are exploratory, not industrial pass/fail limits.",
                "Annotation-defined ROIs must be reported; these evaluations do not establish fully automatic ROI localization.",
                "See run_metadata.json and input_hashes.csv for software/configuration and exact inputs."]
        (run_dir / "REPORT.md").write_text("\n".join(note) + "\n", encoding="utf-8")
    if (run_dir / "sweep_summary.csv").is_file():
        rows = [r for r in read_csv(run_dir / "sweep_summary.csv") if r["setting"].startswith("threshold_")
                and r["cohort"] == "all_fields_with_available_pairs"]
        rows.sort(key=lambda r: float(r["setting"].split("_")[-1]))
        for key, label in (("g_mae", "Paired G MAE"), ("instance_f1_micro", "All-ROI micro instance F1"),
                           ("prediction_undefined_rate", "Fraction with undefined predicted G")):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            valid = [r for r in rows if finite(r[key])]
            ax.plot([float(r["setting"].split("_")[-1]) for r in valid], [float(r[key]) for r in valid], marker="o")
            if key == "g_mae":
                common = [r for r in read_csv(run_dir / "sweep_summary.csv")
                          if r["setting"].startswith("threshold_") and r["cohort"] == "common_G_defined_across_all_settings" and finite(r[key])]
                common.sort(key=lambda r: float(r["setting"].split("_")[-1]))
                if common:
                    ax.lines[0].set_label("Available G pairs at each setting")
                    ax.plot([float(r["setting"].split("_")[-1]) for r in common], [float(r[key]) for r in common],
                            marker="s", linestyle="--", label="Same G-valid fields across all settings")
                    ax.legend(fontsize=8)
            ax.set(xlabel="Boundary threshold", ylabel=label)
            finish(fig, ax, f"F4_threshold_{key}", "Held-out sensitivity (descriptive; no best threshold selected)")
    if (run_dir / "corruption_summary.csv").is_file():
        rows = read_csv(run_dir / "corruption_summary.csv")
        for key, label in (("g_bias", "Signed G disagreement"), ("instance_f1_micro", "All-ROI micro instance F1")):
            fig, ax = plt.subplots(figsize=(8, 5))
            for kind in sorted({r["corruption_kind"] for r in rows}):
                part = sorted([r for r in rows if r["corruption_kind"] == kind and finite(r[key])],
                              key=lambda r: float(r["requested_strength"]))
                if part:
                    ax.plot([float(r["requested_strength"]) for r in part], [float(r[key]) for r in part], marker="o", label=kind)
            ax.set(xlabel="Requested intervention fraction (see realized operations in CSV)", ylabel=label)
            ax.legend(fontsize=8)
            finish(fig, ax, f"F5_corruption_{key}", "Controlled label interventions; repeats are not independent fields")
    return produced
