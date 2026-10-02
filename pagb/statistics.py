"""Field-weighted summaries with specimen-cluster bootstrap.

Resampling units are specimens; all fields from a selected specimen are copied
TOGETHER, including multiplicity. These are conditional CIs for cached predictions,
not uncertainty over model retraining/seeds or a cure for dependent CV training sets.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from .io import finite


def values(rows, key):
    return np.array([float(r[key]) for r in rows if finite(r.get(key))], dtype=float)


def mean(rows, key):
    v = values(rows, key)
    return float(v.mean()) if len(v) else None


def failure_counts(rows: list[dict]) -> dict:
    counts = {"n_total": len(rows), "both_defined": 0, "reference_only_undefined": 0,
              "prediction_only_undefined": 0, "both_undefined": 0}
    for r in rows:
        p, t = finite(r.get("pred_astm_g")), finite(r.get("true_astm_g"))
        key = "both_defined" if p and t else "both_undefined" if not p and not t else "reference_only_undefined" if not t else "prediction_only_undefined"
        counts[key] += 1
    counts["prediction_undefined_total"] = counts["prediction_only_undefined"] + counts["both_undefined"]
    counts["reference_undefined_total"] = counts["reference_only_undefined"] + counts["both_undefined"]
    counts["excluded_from_paired_g"] = len(rows) - counts["both_defined"]
    assert sum(counts[k] for k in ("both_defined", "reference_only_undefined", "prediction_only_undefined", "both_undefined")) == len(rows)
    return counts


def summarize(rows: list[dict]) -> dict:
    failures = failure_counts(rows)
    delta = values(rows, "delta_g")
    ratios = values(rows, "d_plan_ratio")
    ref_count = sum(float(r.get("true_counted_grains", 0)) for r in rows)
    pred_count = sum(float(r.get("pred_counted_grains", 0)) for r in rows)
    tp = sum(int(r.get("instance_all_tp", 0)) for r in rows)
    fp = sum(int(r.get("instance_all_fp", 0)) for r in rows)
    fn = sum(int(r.get("instance_all_fn", 0)) for r in rows)
    paired = [r for r in rows if finite(r.get("count_ratio")) and finite(r.get("measurement_area_ratio"))
              and r["count_ratio"] > 0 and r["measurement_area_ratio"] > 0]
    slope = slope_se = intercept = r2 = None
    if len(paired) >= 3:
        x = np.log([r["count_ratio"] for r in paired])
        y = np.log([r["measurement_area_ratio"] for r in paired])
        dx, dy = x - x.mean(), y - y.mean()
        sxx = float(dx @ dx)
        if sxx > 1e-14:
            slope = float((dx @ dy) / sxx)
            intercept = float(y.mean() - slope * x.mean())
            residual = y - intercept - slope * x
            slope_se = float(np.sqrt((residual @ residual) / (len(x) - 2) / sxx))
            syy = float(dy @ dy)
            r2 = 1 - float(residual @ residual) / syy if syy > 1e-14 else None
    by_group = defaultdict(list)
    for r in rows:
        if finite(r.get("delta_g")):
            by_group[r["group_id"]].append(abs(float(r["delta_g"])))
    # If passed bootstrap copies, group macro statistic collapses duplicates. It is
    # reported descriptively only; cluster_bootstrap intentionally excludes it.
    macro_specimen = float(np.mean([np.mean(v) for v in by_group.values()])) if by_group else None
    return {
        **failures, "n_specimens": len({r["group_id"] for r in rows}),
        "n_unique_fields": len({r["sample_id"] for r in rows}),
        "g_mae": float(np.mean(abs(delta))) if len(delta) else None,
        "g_rmse": float(np.sqrt(np.mean(delta ** 2))) if len(delta) else None,
        "g_bias": float(delta.mean()) if len(delta) else None,
        "g_mae_specimen_macro_descriptive": macro_specimen,
        "instance_f1_field_mean": mean(rows, "instance_all_f1"),
        "instance_f1_micro": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "instance_precision_micro": tp / (tp + fp) if tp + fp else (0.0 if fn else None),
        "instance_recall_micro": tp / (tp + fn) if tp + fn else None,
        "pred_count_total": pred_count, "reference_count_total": ref_count,
        "pooled_count_ratio": pred_count / ref_count if ref_count else None,
        "pred_measured_area_fraction_roi_mean": mean(rows, "pred_measurement_area_fraction_roi"),
        "reference_measured_area_fraction_roi_mean": mean(rows, "true_measurement_area_fraction_roi"),
        "pred_measured_area_fraction_frame_mean": mean(rows, "pred_measurement_area_fraction_frame"),
        "reference_measured_area_fraction_frame_mean": mean(rows, "true_measurement_area_fraction_frame"),
        "geometric_mean_size_ratio": float(np.exp(np.log(ratios[ratios > 0]).mean())) if np.any(ratios > 0) else None,
        "median_absolute_size_deviation": float(np.median(abs(ratios - 1))) if len(ratios) else None,
        "minimum_size_ratio": float(ratios.min()) if len(ratios) else None,
        "maximum_size_ratio": float(ratios.max()) if len(ratios) else None,
        "area_count_slope": slope, "area_count_slope_ols_se_not_cluster_ci": slope_se,
        "area_count_intercept": intercept, "area_count_r2": r2, "n_area_count_pairs": len(paired),
        "prediction_undefined_rate": failures["prediction_undefined_total"] / len(rows) if rows else None,
    }


BOOTSTRAP_KEYS = ("g_mae", "g_rmse", "g_bias", "instance_f1_field_mean", "instance_f1_micro",
                  "pooled_count_ratio", "pred_measured_area_fraction_roi_mean",
                  "reference_measured_area_fraction_roi_mean", "pred_measured_area_fraction_frame_mean",
                  "reference_measured_area_fraction_frame_mean", "geometric_mean_size_ratio",
                  "median_absolute_size_deviation", "area_count_slope", "area_count_intercept", "prediction_undefined_rate")


def cluster_indices(rows: list[dict], rng: np.random.Generator) -> np.ndarray:
    groups = sorted({r["group_id"] for r in rows})
    if not groups:
        return np.array([], dtype=int)
    indices = {g: np.array([i for i, r in enumerate(rows) if r["group_id"] == g], dtype=int) for g in groups}
    sampled = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([indices[groups[i]] for i in sampled])


def contributing_specimens(rows: list[dict], key: str) -> int:
    """A cohort can contain many specimens but only one can support a given metric."""
    def eligible(r):
        if key in ("g_mae", "g_rmse", "g_bias"):
            return finite(r.get("delta_g"))
        if key in ("geometric_mean_size_ratio", "median_absolute_size_deviation"):
            return finite(r.get("d_plan_ratio")) and float(r["d_plan_ratio"]) > 0
        if key in ("area_count_slope", "area_count_intercept"):
            return all(finite(r.get(k)) and float(r[k]) > 0 for k in ("count_ratio", "measurement_area_ratio"))
        if key == "instance_f1_micro":
            return any(float(r.get(k, 0) or 0) > 0 for k in ("instance_all_tp", "instance_all_fp", "instance_all_fn"))
        if key == "pooled_count_ratio":
            return any(float(r.get(k, 0) or 0) > 0 for k in ("pred_counted_grains", "true_counted_grains"))
        source = {"instance_f1_field_mean": "instance_all_f1",
                  "pred_measured_area_fraction_roi_mean": "pred_measurement_area_fraction_roi",
                  "reference_measured_area_fraction_roi_mean": "true_measurement_area_fraction_roi",
                  "pred_measured_area_fraction_frame_mean": "pred_measurement_area_fraction_frame",
                  "reference_measured_area_fraction_frame_mean": "true_measurement_area_fraction_frame"}.get(key)
        return finite(r.get(source)) if source else True
    return len({r["group_id"] for r in rows if eligible(r)})


def cluster_bootstrap(rows: list[dict], n_replicates: int, seed: int) -> dict:
    point = summarize(rows)
    groups = sorted({r["group_id"] for r in rows})
    if n_replicates <= 0 or len(groups) < 2:
        return {"method": "specimen_cluster_percentile", "n_specimens": len(groups), "requested_replicates": n_replicates,
                "status": "disabled" if n_replicates <= 0 else "insufficient_independent_specimens", "intervals": {}}
    rng = np.random.default_rng(seed)
    # Build the cluster index lists once, not once per bootstrap draw.
    indices = [np.array([i for i, r in enumerate(rows) if r["group_id"] == g], dtype=int) for g in groups]
    estimates = {k: [] for k in BOOTSTRAP_KEYS}
    for _ in range(n_replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([indices[i] for i in sampled])
        stat = summarize([rows[int(i)] for i in idx])
        for key in BOOTSTRAP_KEYS:
            if finite(stat[key]):
                estimates[key].append(stat[key])
    intervals = {}
    for key, vals in estimates.items():
        valid_fraction = len(vals) / n_replicates
        n_contributing = contributing_specimens(rows, key)
        interval = np.quantile(vals, [0.025, 0.975]).tolist() if len(vals) >= 2 and n_contributing >= 2 else [None, None]
        intervals[key] = {"estimate": point[key], "lower_95": interval[0], "upper_95": interval[1],
                          "defined_replicates": len(vals), "defined_fraction": valid_fraction,
                          "contributing_specimens": n_contributing,
                          "status": "insufficient_independent_specimens" if n_contributing < 2 else
                                    "insufficient_defined_replicates" if len(vals) < 2 else "estimated",
                          "warning": "conditional_on_defined_replicates" if valid_fraction < 1 else None}
    return {"method": "specimen_cluster_percentile", "requested_replicates": n_replicates,
            "n_specimens": len(groups), "seed": seed,
            "scope": "Fixed cached predictions; excludes retraining/seed uncertainty. Descriptive with overlapping CV training sets.",
            "intervals": intervals}


def strata(rows: list[dict], g_bins: list[float]) -> list[dict]:
    groups = defaultdict(list)
    groups[("all", "all")].extend(rows)
    for r in rows:
        groups[("magnification", str(r.get("magnification_x") or "unknown"))].append(r)
        g = r.get("true_astm_g")
        if not finite(g):
            label = "reference_G_undefined"
        else:
            index = int(np.searchsorted(g_bins, g, side="right"))
            lo = "-inf" if index == 0 else str(g_bins[index - 1])
            hi = "inf" if index == len(g_bins) else str(g_bins[index])
            label = f"[{lo},{hi})"
        groups[("reference_G", label)].append(r)
    return [{"stratum": key[0], "level": key[1], **summarize(value)} for key, value in groups.items()]
