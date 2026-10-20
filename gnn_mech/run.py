"""Command-line entry point: train one config/seed, then measure it.

    python -m gnn_mech.run --config configs/smoke.yaml --seed 0 [--force] [--no-measure] [key=value ...]

Runs resume from `checkpoint.pt` automatically; a run with `metrics.json` is
skipped unless `--force` (training only; missing measurements and prediction files
are still produced). After training, the best-val model's per-graph outputs are
written to `preds_val.npz` / `preds_test.npz` (graph_id, y, logits) for stratified
metrics and bootstrap CIs without retraining.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from .config import Config, load_config
from .results import RunDir
from .train import predict, resolve_device, set_seed, train

PRED_SPLITS = ("val", "test")
_RUN_FILES = ("checkpoint.pt", "metrics.json", "history.json", "best.pt", "measurements.json",
              *(f"preds_{s}.npz" for s in PRED_SPLITS))


def _save_preds(run_dir: RunDir, split: str, preds: dict) -> None:
    tmp = run_dir.file(f"preds_{split}.tmp.npz")
    np.savez(tmp, **preds)
    os.replace(tmp, run_dir.file(f"preds_{split}.npz"))


def run(cfg: Config, force: bool = False, measure: bool = True) -> dict:
    """Train (or skip/resume) and measure one run; returns metrics plus "run_dir"."""
    from .data import load_dataset
    from .models import build_model

    run_dir = RunDir(cfg)
    if force:
        for name in _RUN_FILES:
            run_dir.file(name).unlink(missing_ok=True)

    do_measure = measure and cfg.measure.enabled
    have_preds = all(run_dir.file(f"preds_{s}.npz").exists() for s in PRED_SPLITS)
    if run_dir.is_complete() and have_preds and (not do_measure or run_dir.file("measurements.json").exists()):
        print(f"[skip] already complete: {run_dir.path}")
        return {**run_dir.load_json("metrics.json"), "run_dir": str(run_dir.path)}

    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)
    train_ds, val_ds, test_ds, info = load_dataset(cfg.data)
    model = build_model(cfg.model, info).to(device)

    if run_dir.is_complete():
        print(f"[skip] training complete, predicting/measuring only: {run_dir.path}")
        metrics = run_dir.load_json("metrics.json")
    else:
        metrics = train(cfg, model, train_ds, val_ds, test_ds, info, run_dir)

    model.load_state_dict(run_dir.load_torch("best.pt", map_location=device))
    model.eval()

    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    for split in PRED_SPLITS:
        _save_preds(run_dir, split, predict(model, splits[split], cfg.train.batch_size, device))

    if do_measure and not run_dir.file("measurements.json").exists():
        from .measure import run_all

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
