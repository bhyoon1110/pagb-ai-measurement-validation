from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .io import clean_json

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m pagb", description="Audited PAGB/EAAI experiment tools; no fabricated data or results.")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check archive inputs/environment (missing research data is reported, not synthesized).")
    doctor.add_argument("--workspace", type=Path, default=ROOT)
    doctor.add_argument("--output", type=Path)
    demo = sub.add_parser("demo", help="Create clearly labelled synthetic test inputs.")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=20260928)
    imp = sub.add_parser("import-legacy", help="Adapt existing project manifests/cached predictions; no filename-based specimen guesses.")
    imp.add_argument("--workspace", type=Path, required=True)
    imp.add_argument("--output", type=Path, required=True)
    for name in ("manifest", "folds", "calibration", "prediction-root", "instance-dir"):
        imp.add_argument("--" + name, type=Path)
    imp.add_argument("--model-id", default="unet")
    imp.add_argument("--run-id", default="legacy_unverified")
    for command in ("evaluate", "sweep", "corrupt", "all"):
        sp = sub.add_parser(command)
        sp.add_argument("--manifest", type=Path, required=True)
        sp.add_argument("--output", type=Path, required=True)
        sp.add_argument("--config", type=Path)
        sp.add_argument("--data-root", type=Path)
        sp.add_argument("--allow-unverified-provenance", action="store_true")
        sp.add_argument("--allow-missing-scale", action="store_true")
        if command in ("evaluate", "all"):
            sp.add_argument("--prediction-source", choices=("auto", "cached", "probability"), default="auto")
            sp.add_argument("--selected-parameters", type=Path,
                            help="Inner-only selection JSON bound to this held-out manifest.")
    tune = sub.add_parser("tune", help="Select thresholds on verified same-model INNER caches only.")
    tune.add_argument("--inner-manifest", type=Path, required=True)
    tune.add_argument("--heldout-manifest", type=Path, required=True)
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument("--config", type=Path)
    paper = sub.add_parser("paper-export", help="Export auditable tables/claims; synthetic/unverified runs cannot be certified.")
    paper.add_argument("--run-dir", type=Path, required=True)
    paper.add_argument("--output", type=Path, required=True)
    paper.add_argument("--expected-fields", type=int)
    paper.add_argument("--expected-specimens", type=int)
    paper.add_argument("--require-verified", action="store_true")
    rep = sub.add_parser("report", help="Generate figures from this run's tables, with no hardcoded manuscript results.")
    rep.add_argument("--run-dir", type=Path, required=True)
    cache = sub.add_parser("cache", help="Export OOF and optional inner-validation predictions with checkpoint provenance.")
    cache.add_argument("--workspace", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--model-root", type=Path, required=True)
    cache.add_argument("--config", type=Path)
    for name in ("manifest", "folds", "calibration"):
        cache.add_argument("--" + name, type=Path)
    cache.add_argument("--model-id", default="unet")
    cache.add_argument("--run-id", default="eaai_v3")
    cache.add_argument("--include-inner", action="store_true")
    cache.add_argument("--fold", type=int)
    cache.add_argument("--allow-unverified-checkpoints", action="store_true")
    cache.add_argument("--patch-size", type=int, default=256)
    cache.add_argument("--stride", type=int, default=192)
    cache.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args(argv)
    try:
        if args.command == "doctor":
            from .doctor import doctor
            result = doctor(args.workspace, args.output)
        elif args.command == "demo":
            from .demo import make_demo
            result = {"manifest": str(make_demo(args.output, args.seed)), "synthetic_only": True}
        elif args.command == "import-legacy":
            from .data import import_legacy
            kw = vars(args).copy(); kw.pop("command")
            result = import_legacy(**kw)
        elif args.command == "paper-export":
            from .paper_export import export_paper
            result = export_paper(args.run_dir, args.output, args.expected_fields,
                                  args.expected_specimens, args.require_verified)
        elif args.command == "report":
            from .reports import report
            result = {"figures": report(args.run_dir)}
        else:
            from .config import load_config
            config = load_config(args.config)
            if args.command == "cache":
                from .inference import cache_predictions
                kw = vars(args).copy(); kw.pop("command"); kw["config"] = config
                kw["only_fold"] = kw.pop("fold")
                result = cache_predictions(**kw)
            elif args.command == "tune":
                from .selection import tune
                result = tune(args.inner_manifest, args.heldout_manifest, args.output, config)
            else:
                from .experiments import evaluate, sweep, corruption_experiment
                from .reports import report
                options = {"data_root": args.data_root, "allow_unverified": args.allow_unverified_provenance,
                           "allow_missing_scale": args.allow_missing_scale}
                if args.command == "evaluate":
                    result = evaluate(args.manifest, args.output, config, selected_parameters=args.selected_parameters, prediction_source=args.prediction_source, **options)
                    report(args.output)
                elif args.command == "sweep":
                    result = sweep(args.manifest, args.output, config, **options)
                    report(args.output)
                elif args.command == "corrupt":
                    result = corruption_experiment(args.manifest, args.output, config, **options)
                    report(args.output)
                else:
                    from .io import fresh_output, write_json
                    # Preflight probability presence before the first expensive stage.
                    from .data import load_manifest
                    load_manifest(args.manifest, require_probability=True, **options)
                    fresh_output(args.output)
                    result = {}
                    result["evaluation"] = evaluate(args.manifest, args.output / "evaluation", config,
                                                    selected_parameters=args.selected_parameters, prediction_source=args.prediction_source, **options)
                    report(args.output / "evaluation")
                    result["sweep"] = sweep(args.manifest, args.output / "sweep", config, **options)
                    report(args.output / "sweep")
                    result["corruption"] = corruption_experiment(args.manifest, args.output / "corruption", config, **options)
                    report(args.output / "corruption")
                    from .diagnostics import audit_baseline_consistency
                    result["baseline_consistency"] = audit_baseline_consistency(args.output)
                    write_json(args.output / "completed_stages.json", result)
        print(json.dumps(clean_json(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
