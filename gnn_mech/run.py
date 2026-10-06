"""Command-line entry point: train one config/seed, then measure it.

    python -m gnn_mech.run --config configs/smoke.yaml --seed 0 [--force] [--no-measure] [key=value ...]

Runs resume from `checkpoint.pt` automatically; a run with `metrics.json` is
skipped unless `--force` (training only; missing measurements are still taken).
"""
from __future__ import annotations

import argparse
from typing import Optional

from .config import Config, load_config
from .results import RunDir
from .train import resolve_device, set_seed, train

_RUN_FILES = ("checkpoint.pt", "metrics.json", "history.json", "best.pt", "measurements.json")


def run(cfg: Config, force: bool = False, measure: bool = True) -> dict:
    """Train (or skip/resume) and measure one run; returns metrics plus "run_dir"."""
    from .data import load_dataset
    from .models import build_model

    run_dir = RunDir(cfg)
    if force:
        for name in _RUN_FILES:
            run_dir.file(name).unlink(missing_ok=True)

    do_measure = measure and cfg.measure.enabled
    if run_dir.is_complete() and (not do_measure or run_dir.file("measurements.json").exists()):
        print(f"[skip] already complete: {run_dir.path}")
        return {**run_dir.load_json("metrics.json"), "run_dir": str(run_dir.path)}

    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    train_ds, val_ds, test_ds, info = load_dataset(cfg.data)
    model = build_model(cfg.model, info).to(device)

    if run_dir.is_complete():
        print(f"[skip] training complete, measuring only: {run_dir.path}")
        metrics = run_dir.load_json("metrics.json")
    else:
        metrics = train(cfg, model, train_ds, val_ds, test_ds, info, run_dir)

    model.load_state_dict(run_dir.load_torch("best.pt", map_location=device))
    model.eval()

    if do_measure:
        from .measure import run_all

        splits = {"train": train_ds, "val": val_ds, "test": test_ds}
        if cfg.measure.split not in splits:
            raise ValueError(f"measure.split must be one of {sorted(splits)}, got {cfg.measure.split!r}")
        run_dir.save_json("measurements.json", run_all(model, splits[cfg.measure.split], cfg.measure, device))

    print(f"[done] {run_dir.path}")
    return {**metrics, "run_dir": str(run_dir.path)}


def parse_args(argv: Optional[list[str]] = None) -> tuple[argparse.Namespace, Config]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--seed", type=int, default=None, help="overrides cfg.seed")
    parser.add_argument("--force", action="store_true", help="discard existing results and retrain")
    parser.add_argument("--no-measure", action="store_true", help="skip mechanistic measurements")
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. model.alpha=0.25")
    args = parser.parse_args(argv)
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    return args, load_config(args.config, overrides)


def main(argv: Optional[list[str]] = None) -> dict:
    args, cfg = parse_args(argv)
    return run(cfg, force=args.force, measure=not args.no_measure)


if __name__ == "__main__":
    main()
