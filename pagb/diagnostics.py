"""Signed count/area diagnostics and consistency checks, NOT causal identification.

Positive attenuation means smaller *disagreement with the annotation* than the
count-only diagnostic. Negative values are retained (amplification). Neither
sign establishes correctness, physical accuracy, or a valid fixed-area estimator.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from .io import finite, read_csv, write_csv, write_json
from .measurement import G_COEFFICIENT


def coupling_summary(rows: list[dict]) -> dict:
    pairs = [r for r in rows if finite(r.get("fixed_reference_area_delta_g"))
             and finite(r.get("coupled_delta_g_dimensionless"))]
    c = np.array([float(r["fixed_reference_area_delta_g"]) for r in pairs])
    d = np.array([float(r["coupled_delta_g_dimensionless"]) for r in pairs])
    a = d - c
    attenuation = np.abs(c) - np.abs(d)
    eps = 1e-10
    # Aggregate sign cancellation is distinct from within-field count/area cancellation.
    absolute_pairs = [r for r in rows if finite(r.get("delta_g"))
                      and finite(r.get("d_plan_ratio")) and float(r["d_plan_ratio"]) > 0]
    signed = np.array([float(r["delta_g"]) for r in absolute_pairs])
    ratios = np.array([float(r["d_plan_ratio"]) for r in absolute_pairs])
    implied = 10 ** (-float(signed.mean()) / (2 * G_COEFFICIENT)) if len(signed) else None
    observed = float(np.exp(np.mean(np.log(ratios)))) if len(ratios) else None
    out = {
        "n_all_fields": len(rows), "n_relative_evaluable_fields": len(pairs),
        "n_relative_evaluable_specimens": len({r["group_id"] for r in pairs}),
        "n_absolute_G_and_size_pairs": len(absolute_pairs),
        "n_attenuation": int(np.count_nonzero(attenuation > eps)),
        "n_amplification": int(np.count_nonzero(attenuation < -eps)),
        "n_unchanged_absolute_disagreement": int(np.count_nonzero(np.abs(attenuation) <= eps)),
        "n_count_area_opposing_signs": int(np.count_nonzero(c * a < -eps)),
        "n_coupled_sign_reversals": int(np.count_nonzero(c * d < -eps)),
        "mean_absolute_count_only_delta_G": float(np.mean(abs(c))) if len(c) else None,
        "mean_absolute_coupled_delta_G": float(np.mean(abs(d))) if len(d) else None,
        "mean_absolute_G_attenuation": float(attenuation.mean()) if len(attenuation) else None,
        "median_absolute_G_attenuation": float(np.median(attenuation)) if len(attenuation) else None,
        "fraction_with_attenuation": float(np.mean(attenuation > eps)) if len(attenuation) else None,
        "geometric_size_ratio_same_G_cohort": observed,
        "geometric_size_ratio_implied_by_signed_G_bias": implied,
        "aggregate_identity_residual": observed - implied if observed is not None else None,
        "interpretation": "Algebraic decomposition, not proof of physical error or causality. "
                          "Fixed-reference-area is an annotation-dependent diagnostic, not a deployable ASTM alternative. "
                          "Slope=1 alone is insufficient for full compensation; intercept and field residuals matter.",
    }
    assert out["n_attenuation"] + out["n_amplification"] + out["n_unchanged_absolute_disagreement"] == len(pairs)
    return out


def paired_attenuation_ci(rows: list[dict], n_replicates: int, seed: int) -> dict:
    valid = [r for r in rows if finite(r.get("fixed_reference_area_delta_g"))
             and finite(r.get("coupled_delta_g_dimensionless"))]
    groups = sorted({r["group_id"] for r in rows})
    contributing = {r["group_id"] for r in valid}
    buckets = {g: [abs(float(r["fixed_reference_area_delta_g"])) - abs(float(r["coupled_delta_g_dimensionless"]))
                   for r in valid if r["group_id"] == g] for g in groups}
    estimates = [v for g in groups for v in buckets[g]]
    result = {"method": "paired_specimen_cluster_percentile", "requested_replicates": n_replicates,
              "seed": seed, "n_all_specimens": len(groups), "n_contributing_specimens": len(contributing),
              "n_paired_fields": len(valid), "estimate": float(np.mean(estimates)) if estimates else None,
              "lower_95": None, "upper_95": None, "defined_replicates": 0,
              "status": "not_estimated", "scope": "Fixed predictions; not training-seed or annotation uncertainty."}
    if n_replicates <= 0 or len(contributing) < 2:
        result["status"] = "disabled" if n_replicates <= 0 else "insufficient_independent_specimens"
        return result
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_replicates):
        sample = [v for j in rng.integers(0, len(groups), len(groups)) for v in buckets[groups[j]]]
        if sample:
            boot.append(float(np.mean(sample)))
    if len(boot) >= 2:
        result["lower_95"], result["upper_95"] = map(float, np.quantile(boot, [.025, .975]))
        result["status"] = "estimated" if len(boot) == n_replicates else "conditional_on_defined_replicates"
    result["defined_replicates"] = len(boot)
    return result


def export_coupling_diagnostics(rows: list[dict], output: Path, config: dict) -> None:
    summary = coupling_summary(rows)
    summary["paired_attenuation_ci"] = paired_attenuation_ci(
        rows, config["bootstrap_replicates"], config["bootstrap_seed"])
    write_json(output / "coupling_diagnostic_summary.json", summary)
    write_csv(output / "coupling_cases.csv", [{
        "sample_id": r["sample_id"], "group_id": r["group_id"],
        "count_component_delta_g": r.get("count_component_delta_g"),
        "area_component_delta_g": r.get("area_component_delta_g"),
        "coupled_delta_g": r.get("coupled_delta_g_dimensionless"),
        "absolute_G_attenuation": r.get("absolute_g_attenuation"),
        "instance_f1_all_roi": r.get("instance_all_f1"),
        "case": ("undefined" if not finite(r.get("absolute_g_attenuation")) else
                 "attenuation" if float(r["absolute_g_attenuation"]) > 1e-10 else
                 "amplification" if float(r["absolute_g_attenuation"]) < -1e-10 else "unchanged"),
    } for r in rows])


def audit_baseline_consistency(run_dir: Path) -> dict:
    """Detect cached/selected evaluation != fixed-probability sweep baseline.

    A difference may be intentional (e.g. inner-selected thresholds). It is recorded,
    not silently pooled or called the same experiment.
    """
    evaluation = read_csv(run_dir / "evaluation/per_field_metrics.csv")
    sweep = [r for r in read_csv(run_dir / "sweep/sweep_per_field.csv") if r["setting"] == "baseline"]
    e, s = ({r["sample_id"]: r for r in table} for table in (evaluation, sweep))
    if len(e) != len(evaluation) or len(s) != len(sweep) or set(e) != set(s):
        raise ValueError("Evaluation/sweep baseline fields must match exactly, without duplicates.")
    keys = ("pred_counted_grains", "pred_measurement_px", "delta_g", "instance_all_f1")
    details = []
    for sid in sorted(e):
        differences = []
        for key in keys:
            x, y = e[sid].get(key), s[sid].get(key)
            equal = math.isclose(float(x), float(y), rel_tol=1e-10, abs_tol=1e-10) if finite(x) and finite(y) else not finite(x) and not finite(y)
            if not equal:
                differences.append(key)
        for key in ("prediction_label_sha256", "reference_label_sha256"):
            if e[sid].get(key) != s[sid].get(key):
                differences.append(key)
        details.append({"sample_id": sid, "group_id": e[sid]["group_id"],
                        "evaluation_prediction_source": e[sid].get("prediction_source"),
                        "different": bool(differences), "different_quantities": ";".join(differences)})
    result = {"n_fields": len(details), "n_different": sum(r["different"] for r in details),
              "note": "Different cached or inner-selected labels are not the fixed-setting sweep baseline. "
                      "Use --prediction-source probability for the fixed-setting comparison. Selected settings may intentionally differ."}
    write_csv(run_dir / "evaluation_sweep_consistency.csv", details)
    write_json(run_dir / "evaluation_sweep_consistency.json", result)
    return result
