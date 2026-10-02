"""Auditable experiments over one locked prediction/reference manifest."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from . import __version__
from .config import load_config
from .data import load_manifest, load_field, PATH_COLUMNS
from .io import write_csv, write_json, sha256, stable_seed, environment, fresh_output, finite
from .measurement import measure, counterfactual, measurement_masks
from .metrics import match_instances, boundary_map, boundary_f1
from .postprocess import Postprocess, watershed_labels
from .statistics import summarize, cluster_bootstrap, strata, failure_counts
from .corruptions import corrupt


def score_field(row: dict, reference: np.ndarray, prediction: np.ndarray, roi: np.ndarray,
                config: dict, include_boundary: bool = True) -> tuple[dict, list[dict]]:
    protocol = config["measurement_protocol"]
    p = measure(prediction, roi, row["um_per_pixel"], protocol)
    t = measure(reference, roi, row["um_per_pixel"], protocol)
    result = {key: row.get(key) for key in ("sample_id", "group_id", "fold", "magnification_x", "um_per_pixel",
              "model_id", "run_id", "evaluation_role", "provenance_status", "roi_origin", "model_outer_fold", "calibration_provenance_status")}
    result.update({f"pred_{k}": v for k, v in p.items()})
    result.update({f"true_{k}": v for k, v in t.items()})
    result.update(counterfactual(p, t))
    for name, labels in (("prediction", prediction), ("reference", reference)):
        scoped = np.where(roi, labels, 0).astype("<i8")
        result[name + "_label_sha256"] = hashlib.sha256(str(scoped.shape).encode() + scoped.tobytes()).hexdigest()
    matches = []
    # This is intentionally BEFORE border deletion and over the entire common ROI.
    for threshold in config["iou_thresholds"]:
        matching = match_instances(reference, prediction, roi, threshold)
        prefix = "instance_all_" if threshold == 0.5 else f"instance_iou_{threshold:g}_"
        result.update({prefix + k: v for k, v in matching.items() if k != "matches"})
        if threshold == 0.5:
            matches = [{"sample_id": row["sample_id"], **m} for m in matching["matches"]]
    # Supplementary post-deletion comparison. Never crop to the intersection of survivors.
    pm = measurement_masks(prediction, roi, protocol)
    tm = measurement_masks(reference, roi, protocol)
    kept_prediction = np.where(pm["interiors"], pm["labels"], 0)
    kept_reference = np.where(tm["interiors"], tm["labels"], 0)
    kept = match_instances(kept_reference, kept_prediction, roi, 0.5)
    result.update({"instance_kept_" + k: v for k, v in kept.items() if k != "matches"})
    if include_boundary:
        rb, pb = boundary_map(reference, roi), boundary_map(prediction, roi)
        for tolerance in config["boundary_tolerances_px"]:
            b = boundary_f1(rb, pb, roi, tolerance)
            for key in ("f1", "precision", "recall"):
                result[f"boundary_{key}_{tolerance:g}px"] = b[key]
        for tolerance in config["boundary_tolerances_um"]:
            b = boundary_f1(rb, pb, roi, tolerance / row["um_per_pixel"]) if row["um_per_pixel"] else {"f1": None}
            result[f"boundary_f1_{tolerance:g}um"] = b["f1"]
    return result, matches


def _provenance(manifest: Path, rows: list[dict], config: dict, output: Path, command: str) -> None:
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    inputs = []
    cached = {}
    for row in rows:
        for key in PATH_COLUMNS:
            path = row.get(key)
            if path:
                if str(path) not in cached:
                    cached[str(path)] = sha256(path)
                inputs.append({"sample_id": row["sample_id"], "input_role": key,
                               "path": str(path), "sha256": cached[str(path)]})
    write_csv(output / "input_hashes.csv", inputs)
    write_json(output / "run_metadata.json", {
        "toolkit_version": __version__, "command": command,
        "manifest_path": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
        "toolkit_source_sha256": source_hash.hexdigest(), "environment": environment(), "config": config,
        "n_input_fields": len(rows), "n_input_specimens": len({r["group_id"] for r in rows}),
        "model_id": rows[0]["model_id"], "run_id": rows[0]["run_id"],
        "provenance_statuses": sorted({r["provenance_status"] for r in rows}),
        "evaluation_roles": sorted({r["evaluation_role"] for r in rows}),
        "roi_origins": sorted({r["roi_origin"] for r in rows}),
        "synthetic_only": all(r["provenance_status"] == "synthetic_demo" for r in rows),
        "interpretation": "Fixed-area variants are diagnostics, NOT certified alternatives or evidence of greater accuracy. G failures remain in the denominator table.",
    })


def _diagnostics(rows: list[dict], config: dict) -> list[dict]:
    valid = [r for r in rows if finite(r["abs_delta_g"]) and finite(r["instance_all_f1"])]
    results = []
    for dg in config["diagnostic_g_tolerances"]:
        for f1 in config["diagnostic_f1_thresholds"]:
            good_g = [r for r in valid if r["abs_delta_g"] <= dg]
            discordant = [r for r in good_g if r["instance_all_f1"] < f1]
            results.append({"g_tolerance": dg, "f1_cutoff": f1,
                            "cutoff_status": "exploratory_diagnostic_not_industrial_acceptance_limit",
                            "n_all_fields": len(rows), "n_paired_defined_g_and_f1": len(valid),
                            "n_small_g_error": len(good_g), "n_small_g_error_low_f1": len(discordant),
                            "fraction_of_small_g_error_fields": len(discordant) / len(good_g) if good_g else None,
                            "fraction_of_paired_evaluable_fields": len(discordant) / len(valid) if valid else None})
    return results


def evaluate(manifest: Path, output: Path, config: dict, selected_parameters: Path | None = None,
             prediction_source: str = "auto", **options) -> dict:
    if prediction_source not in ("auto", "cached", "probability"):
        raise ValueError("prediction_source must be auto, cached or probability.")
    if selected_parameters and prediction_source == "cached":
        raise ValueError("Inner-selected postprocessing requires probability reprocessing, not cached labels.")
    rows = load_manifest(manifest, require_probability=selected_parameters is not None or prediction_source == "probability", **options)
    if prediction_source == "cached" and any(not r.get("prediction_labels_path") for r in rows):
        raise ValueError("prediction_source=cached requires instance labels for every field.")
    config = {**config, "evaluation_prediction_source": prediction_source}
    selected = None
    if selected_parameters:
        selected = json.loads(selected_parameters.read_text(encoding="utf-8"))
        if selected.get("selection_status") != "inner_only_same_outer_model":
            raise ValueError("Selected parameters do not have inner-only provenance.")
        if selected.get("heldout_manifest_sha256") != sha256(manifest):
            raise ValueError("Selected parameters are bound to a different held-out manifest.")
        if selected.get("measurement_protocol") != config["measurement_protocol"] or selected.get("reference_postprocessing") != config["reference_postprocessing"]:
            raise ValueError("Tuning/evaluation measurement or reference protocol mismatch.")
        if any(r["provenance_status"] != "checkpoint_verified" or r["evaluation_role"] != "held_out" for r in rows):
            raise ValueError("Selected-parameter evaluation requires verified held-out records.")
        if any(sha256(r["provenance_path"]) != selected.get("heldout_provenance_sha256", {}).get(r["sample_id"]) for r in rows):
            raise ValueError("Held-out cache provenance changed after inner selection.")
        config = {**config, "postprocessing_selection_status": "inner_only_same_outer_model",
                  "selected_parameters_sha256": sha256(selected_parameters)}
    if any(r["evaluation_role"] not in ("held_out", "synthetic_demo", "external") for r in rows):
        raise ValueError("Main evaluation requires held_out/external fields, not inner-validation fields.")
    fresh_output(output)
    _provenance(manifest, rows, config, output, "evaluate")
    ref_params = Postprocess(**config["reference_postprocessing"])
    pred_params = Postprocess(**config["prediction_postprocessing"])
    scored, all_matches, measurement_audits, conventions, eligibility = [], [], [], [], []
    for row in rows:
        roi, ref, pred, prob = load_field(row, ref_params)
        active_params = pred_params
        if selected is not None:
            key = str(row["model_outer_fold"])
            if key not in selected["selected_per_outer_fold"]:
                raise ValueError(f"Missing selected parameters for outer fold {key}")
            active_params = Postprocess(**selected["selected_per_outer_fold"][key]).validate()
            pred = None
        cached_settings = None
        if pred is not None and row.get("provenance_path"):
            cached_settings = json.loads(row["provenance_path"].read_text(encoding="utf-8")).get("prediction_postprocessing")
        if prediction_source == "probability":
            pred = None
        if pred is not None and cached_settings and cached_settings != active_params.to_dict() and prediction_source == "auto":
            raise ValueError("Cached prediction postprocessing differs from requested settings. "
                             "Use --prediction-source probability to apply settings, or cached for explicit replay.")
        if pred is None:
            pred, pp_audit = watershed_labels(prob, roi, active_params, return_audit=True)
        else:
            pp_audit = {"topology_mode": "cached_instance_labels", "seed_count": None,
                        "removed_small_labels": None, "topology_changed_px": None}
        score, matches = score_field(row, ref, pred, roi, config)
        score.update({"postprocessing_" + k: v for k, v in pp_audit.items()})
        active_record = active_params.to_dict() if pp_audit["topology_mode"] != "cached_instance_labels" else (cached_settings or {})
        score.update({"active_" + k: active_record.get(k) for k in active_params.to_dict()})
        score["prediction_source"] = "cached_instance_labels" if pp_audit["topology_mode"] == "cached_instance_labels" else "probability_reprocessed"
        score["cached_postprocessing_matches_requested"] = (cached_settings == active_params.to_dict()) if cached_settings else None
        scored.append(score)
        all_matches.extend(matches)
        eligibility.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                            "pred_g_status": score["pred_g_status"], "true_g_status": score["true_g_status"],
                            "pred_zero_count": score["pred_counted_grains"] == 0,
                            "true_zero_count": score["true_counted_grains"] == 0,
                            "paired_g_included": finite(score["delta_g"]),
                            "pred_count": score["pred_counted_grains"], "true_count": score["true_counted_grains"],
                            "pred_measurement_px": score["pred_measurement_px"], "true_measurement_px": score["true_measurement_px"]})
        for side, arr in (("prediction", pred), ("reference", ref)):
            old, new = (measure(arr, roi, row["um_per_pixel"], p) for p in ("legacy_v1", "strict_v2"))
            measurement_audits.append({"sample_id": row["sample_id"], "side": side,
                                      "legacy_area_px": old["measurement_px"], "strict_area_px": new["measurement_px"],
                                      "delta_area_px": new["measurement_px"] - old["measurement_px"],
                                      "legacy_g": old["astm_g"], "strict_g": new["astm_g"],
                                      "strict_minus_legacy_g": new["astm_g"] - old["astm_g"] if new["astm_g"] is not None and old["astm_g"] is not None else None})
        base = measure(pred, roi, row["um_per_pixel"], config["measurement_protocol"])
        for key, name in (("astm_g", "coupled_baseline"),
                          ("numerator_only_half_count_g", "hybrid_numerator_only_NOT_Jeffries"),
                          ("fixed_roi_count_only_g", "fixed_roi_count_only_diagnostic"),
                          ("jeffries_style_roi_g", "Jeffries_style_count_AND_ROI_area")):
            g, bg = base[key], base["astm_g"]
            conventions.append({"sample_id": row["sample_id"], "group_id": row["group_id"], "variant": name,
                                "g": g, "baseline_g": bg, "delta_g": g - bg if g is not None and bg is not None else None,
                                "g_defined": g is not None,
                                "numerator": base["jeffries_count"] if "half_count" in key or key.startswith("jeffries") else base["counted_grains"],
                                "denominator_mm2": base["roi_area_mm2"] if key in ("fixed_roi_count_only_g", "jeffries_style_roi_g") else base["area_mm2"]})
        print(f"evaluate {len(scored)}/{len(rows)} {row['sample_id']}: G={score['pred_g_status']}, F1={score['instance_all_f1']}", flush=True)
    write_csv(output / "per_field_metrics.csv", scored)
    write_csv(output / "field_eligibility.csv", eligibility)
    if config["save_matches"]:
        write_csv(output / "instance_matches_iou05.csv", all_matches,
                  ["sample_id", "reference_id", "prediction_id", "iou"])
    write_csv(output / "measurement_protocol_audit.csv", measurement_audits)
    write_csv(output / "convention_variants.csv", conventions)
    write_csv(output / "diagnostic_discordance_grid.csv", _diagnostics(scored, config))
    write_csv(output / "stratified_summary.csv", strata(scored, config["reference_g_bins"]))
    summary = summarize(scored)
    write_json(output / "summary.json", summary)
    write_json(output / "failure_summary.json", failure_counts(scored))
    write_json(output / "bootstrap_ci.json", cluster_bootstrap(scored, config["bootstrap_replicates"], config["bootstrap_seed"]))
    cf_keys = ["sample_id", "group_id", "count_ratio", "measurement_area_ratio", "count_component_delta_g",
               "area_component_delta_g", "coupled_delta_g_dimensionless", "fixed_reference_area_delta_g",
               "common_roi_both_sides_delta_g", "delta_g", "abs_delta_g", "d_plan_ratio",
               "instance_all_f1", "pred_g_status", "true_g_status", "decomposition_residual_g",
               "absolute_g_attenuation", "count_area_opposing_signs", "count_area_sign_reversal", "common_roi_verified"]
    write_csv(output / "counterfactual_per_field.csv", [{k: r.get(k) for k in cf_keys} for r in scored])
    from .diagnostics import export_coupling_diagnostics
    export_coupling_diagnostics(scored, output, config)
    from .paper_export import seal_result_tables
    seal_result_tables(output)
    return summary


def settings(config: dict) -> list[tuple[str, Postprocess]]:
    base = Postprocess(**config["prediction_postprocessing"])
    candidates = [("baseline", base)]
    candidates += [(f"threshold_{t:g}", replace(base, boundary_threshold=t)) for t in config["thresholds"]]
    candidates += [(f"min_distance_{d}", replace(base, min_distance=d)) for d in config["sweep_min_distances"]]
    candidates += [(f"min_area_{a}", replace(base, min_grain_area=a)) for a in config["sweep_min_grain_areas"]]
    candidates += [(f"topology_{t}", replace(base, topology_mode=t)) for t in config["sweep_topology_modes"]]
    for _, p in candidates:
        p.validate()
    return candidates


def sweep(manifest: Path, output: Path, config: dict, **options) -> dict:
    rows = load_manifest(manifest, require_probability=True, **options)
    fresh_output(output)
    _provenance(manifest, rows, config, output, "sweep_descriptive_NO_best_threshold_selection")
    candidates = settings(config)
    scored = []
    by_setting = defaultdict(list)
    for i, row in enumerate(rows, 1):
        roi, ref, _, prob = load_field(row, Postprocess(**config["reference_postprocessing"]))
        for name, params in candidates:
            pred, audit = watershed_labels(prob, roi, params, return_audit=True)
            score, _ = score_field(row, ref, pred, roi, config, include_boundary=False)
            score.update(setting=name, **params.to_dict(), **{"postprocessing_" + k: v for k,v in audit.items()})
            scored.append(score)
            by_setting[name].append(score)
        print(f"sweep {i}/{len(rows)} {row['sample_id']}", flush=True)
    # A variable denominator must never masquerade as a performance improvement.
    common_ids = set.intersection(*[{r["sample_id"] for r in subset if finite(r["delta_g"])} for subset in by_setting.values()])
    summary = [{"setting": name, "cohort": "all_fields_with_available_pairs", **summarize(subset)} for name,subset in by_setting.items()]
    summary += [{"setting": name, "cohort": "common_G_defined_across_all_settings",
                 **summarize([r for r in subset if r["sample_id"] in common_ids])} for name,subset in by_setting.items()]
    comparisons = []
    base = {r["sample_id"]: r for r in by_setting["baseline"]}
    for name, subset in by_setting.items():
        for r in subset:
            b = base[r["sample_id"]]
            comparisons.append({"sample_id": r["sample_id"], "group_id": r["group_id"], "setting": name,
                                "both_setting_and_baseline_G_defined": finite(r["delta_g"]) and finite(b["delta_g"]),
                                "abs_g_error_difference_vs_baseline": abs(r["delta_g"]) - abs(b["delta_g"]) if finite(r["delta_g"]) and finite(b["delta_g"]) else None,
                                "instance_f1_difference_vs_baseline": r["instance_all_f1"] - b["instance_all_f1"] if finite(r["instance_all_f1"]) and finite(b["instance_all_f1"]) else None})
    write_csv(output / "sweep_per_field.csv", scored)
    write_csv(output / "sweep_summary.csv", summary)
    write_csv(output / "paired_setting_comparisons.csv", comparisons)
    write_json(output / "common_cohort.json", {"n_fields": len(common_ids), "sample_ids": sorted(common_ids),
                                               "note": "Report both available-pair and fixed-cohort results. No optimum is selected on held-out fields."})
    # Paired specimen bootstrap differences, not independently bootstrapped arms.
    rng = np.random.default_rng(config["bootstrap_seed"])
    groups = sorted({r["group_id"] for r in rows})
    delta_ci = []
    for name in by_setting:
        subset = [r for r in comparisons if r["setting"] == name]
        for metric in ("abs_g_error_difference_vs_baseline", "instance_f1_difference_vs_baseline"):
            vals = [r for r in subset if finite(r[metric])]
            buckets = {g: [r[metric] for r in vals if r["group_id"] == g] for g in groups}
            boot = []
            contributing = {r["group_id"] for r in vals}
            if len(contributing) >= 2:
                for _ in range(config["bootstrap_replicates"]):
                    sample = [v for j in rng.integers(0, len(groups), len(groups)) for v in buckets[groups[j]]]
                    if sample:
                        boot.append(float(np.mean(sample)))
            lo, hi = np.quantile(boot, [0.025, 0.975]) if len(boot) >= 2 else (None, None)
            delta_ci.append({"setting": name, "metric": metric, "n_defined_pairs": len(vals),
                             "estimate": float(np.mean([r[metric] for r in vals])) if vals else None,
                             "lower_95": lo, "upper_95": hi, "defined_replicates": len(boot),
                             "n_contributing_specimens": len(contributing),
                             "ci_status": "estimated" if len(boot) >= 2 else "insufficient_independent_specimens_or_replicates"})
    write_csv(output / "paired_setting_bootstrap_ci.csv", delta_ci)
    return {"n_fields": len(rows), "n_settings": len(candidates), "n_common_g_fields": len(common_ids)}


def corruption_experiment(manifest: Path, output: Path, config: dict, **options) -> dict:
    rows = load_manifest(manifest, **options)
    fresh_output(output)
    _provenance(manifest, rows, config, output, "controlled_reference_label_interventions")
    scored, by_setting = [], defaultdict(list)
    for i, row in enumerate(rows, 1):
        roi, ref, _, _ = load_field(row, Postprocess(**config["reference_postprocessing"]))
        for kind in config["corruption_kinds"]:
            for strength in config["corruption_strengths"]:
                # Replicate even a zero-strength control to retain a balanced design.
                for rep in range(config["corruption_repeats"]):
                    rng = np.random.default_rng(stable_seed(config["corruption_seed"], row["sample_id"], kind, strength, rep))
                    pred, audit = corrupt(ref, roi, kind, strength, rng, min_piece_area=1)
                    score, _ = score_field(row, ref, pred, roi, config, include_boundary=False)
                    score.update(audit, repeat=rep)
                    score["input_model_id"] = score["model_id"]
                    score["model_id"] = "reference_label_intervention_NOT_neural_prediction"
                    scored.append(score)
                    by_setting[(kind, strength)].append(score)
                    if i == 1 and rep == 0 and strength == max(config["corruption_strengths"]):
                        example = output / "intervention_examples"
                        example.mkdir(exist_ok=True)
                        np.save(example / f"{kind}_reference.npy", ref, allow_pickle=False)
                        np.save(example / f"{kind}_prediction.npy", pred, allow_pickle=False)
                        np.save(example / "roi.npy", roi, allow_pickle=False)
        print(f"corrupt {i}/{len(rows)} {row['sample_id']}", flush=True)
    summaries = []
    for (kind, strength), subset in by_setting.items():
        summaries.append({"corruption_kind": kind, "requested_strength": strength,
                          "mean_realized_operations": float(np.mean([r["realized_operations"] for r in subset])),
                          "mean_realized_fraction": float(np.mean([r["realized_operations_per_initial_interior_grain"] for r in subset])),
                          "minimum_realized_operations": min(r["realized_operations"] for r in subset),
                          "maximum_realized_operations": max(r["realized_operations"] for r in subset),
                          "n_repeats_per_field": config["corruption_repeats"],
                          "interpretation": "Monte_Carlo_label_intervention_not_independent_real_images",
                          **summarize(subset)})
    write_csv(output / "corruption_per_field.csv", scored)
    write_csv(output / "corruption_summary.csv", summaries)
    write_json(output / "interpretation.json", {
        "reference_status": "Author-defined annotation, not physical ground truth.",
        "selection_controls": "Random/size-biased object erasure is not a realistic model of a broken neural boundary. It distinguishes selection from topological errors.",
        "repetition_unit": "Multiple interventions on the same field are NOT independent experimental fields.",
        "realized_operations": "Adjacency/geometry may prevent reaching requested strength; use realized counts.",
        "n_actual_fields": len(rows), "n_actual_specimens": len({r["group_id"] for r in rows})})
    return {"n_actual_fields": len(rows), "n_intervention_records": len(scored)}
