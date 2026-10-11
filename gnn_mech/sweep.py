"""Sweep runner: run a list of (config, overrides, seeds) entries in order.

    python -m gnn_mech.sweep configs/sweeps/phase2.yaml [--dry-run] [--no-measure] [key=value ...]

Sweep file (YAML):

    measure_minutes: 5            # optional: estimated measurement time per run
    runs:
      - name: local               # optional label
        config: configs/peptides_func_hybrid.yaml
        overrides: [model.alpha=0.0]
        seeds: [0, 1, 2]
        est_minutes: 10           # optional: estimated training minutes per seed (GPU)

Trailing `key=value` overrides apply to every run (e.g. `out_dir=/content/drive/...`).
Every run goes through `gnn_mech.run.run`, so finished runs are skipped and
interrupted ones resume from `checkpoint.pt`; after a Colab disconnect, re-running
the same command continues where it stopped. `--dry-run` prints the run list, each
run's status and the estimated remaining hours without training anything.
A failing run is reported and the sweep continues; the exit status is 1 if any failed.
"""
from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .config import Config, load_config

REPO = Path(__file__).resolve().parents[1]


@dataclass
class SweepRun:
    name: str
    config_path: str
    overrides: list[str]
    seed: int
    est_minutes: float
    cfg: Config


def _resolve_config(path: str, sweep_file: Path) -> str:
    """Config path as given, else relative to the sweep file, else to the repo root."""
    for cand in (Path(path), sweep_file.parent / path, REPO / path):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"config {path!r} not found (sweep file {sweep_file})")


def load_sweep(path: str, overrides: Optional[list[str]] = None) -> tuple[list[SweepRun], float]:
    """Expand a sweep file into one SweepRun per (entry, seed); returns (runs, measure_minutes)."""
    sweep_file = Path(path)
    with open(sweep_file) as f:
        spec = yaml.safe_load(f) or {}
    entries = spec.get("runs") or []
    if not entries:
        raise ValueError(f"sweep {path} has no runs")
    runs: list[SweepRun] = []
    for i, e in enumerate(entries):
        unknown = set(e) - {"name", "config", "overrides", "seeds", "est_minutes"}
        if unknown:
            raise KeyError(f"unknown keys in sweep entry {i}: {sorted(unknown)}")
        cfg_path = _resolve_config(e["config"], sweep_file)
        entry_over = [str(o) for o in (e.get("overrides") or [])]
        for seed in e.get("seeds", [0]):
            all_over = entry_over + list(overrides or []) + [f"seed={int(seed)}"]
            cfg = load_config(cfg_path, all_over)
            runs.append(SweepRun(name=str(e.get("name", Path(cfg_path).stem)), config_path=cfg_path,
                                 overrides=all_over, seed=int(seed),
                                 est_minutes=float(e.get("est_minutes", 0.0)), cfg=cfg))
    return runs, float(spec.get("measure_minutes", 0.0))


def _run_path(cfg: Config) -> Path:
    """Same folder as RunDir(cfg).path, without creating it."""
    return Path(cfg.out_dir) / cfg.exp_name / cfg.hash() / f"seed{cfg.seed}"


def run_status(cfg: Config, measure: bool = True) -> str:
    """"done", "partial" (checkpoint or metrics present) or "todo"; never creates folders."""
    path = _run_path(cfg)
    if not path.exists():
        return "todo"
    files = {p.name for p in path.iterdir()}
    need = {"metrics.json", "preds_val.npz", "preds_test.npz"}
    if measure and cfg.measure.enabled:
        need.add("measurements.json")
    if need <= files:
        return "done"
    return "partial" if {"checkpoint.pt", "metrics.json"} & files else "todo"


def describe(runs: list[SweepRun], measure_minutes: float, measure: bool = True) -> str:
    lines, remaining = [], 0.0
    for i, r in enumerate(runs):
        status = run_status(r.cfg, measure)
        est = r.est_minutes + (measure_minutes if measure and r.cfg.measure.enabled else 0.0)
        if status != "done":
            remaining += est
        over = " ".join(o for o in r.overrides if not o.startswith("seed="))
        lines.append(f"[{i:2d}] {status:7s} {r.name:12s} seed {r.seed}  {r.cfg.exp_name}/{r.cfg.hash()}"
                     f"  ~{est:.0f} min  {over}")
    n_done = sum(run_status(r.cfg, measure) == "done" for r in runs)
    lines.append(f"{len(runs)} runs, {n_done} done; estimated remaining {remaining / 60:.1f} h")
    return "\n".join(lines)


def run_sweep(runs: list[SweepRun], measure: bool = True) -> dict:
    """Run every SweepRun in order; returns {"done": [...], "failed": [...]} with run dirs / errors."""
    from .run import run

    done, failed = [], []
    for i, r in enumerate(runs):
        print(f"\n=== [{i + 1}/{len(runs)}] {r.name} seed {r.seed} ({r.cfg.exp_name}/{r.cfg.hash()}) ===")
        try:
            done.append(run(r.cfg, measure=measure)["run_dir"])
        except Exception as e:  # keep going: one bad run should not stop a Colab session
            traceback.print_exc()
            failed.append({"name": r.name, "seed": r.seed, "error": repr(e), "run_dir": str(_run_path(r.cfg))})
    print(f"\nsweep finished: {len(done)} ok, {len(failed)} failed")
    for f in failed:
        print(f"  FAILED {f['name']} seed {f['seed']}: {f['error']}")
    return {"done": done, "failed": failed}


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", help="path to a sweep YAML")
    parser.add_argument("--dry-run", action="store_true", help="print the run list and estimated hours only")
    parser.add_argument("--no-measure", action="store_true", help="skip mechanistic measurements")
    parser.add_argument("overrides", nargs="*", help="dotted overrides applied to every run, e.g. out_dir=...")
    args = parser.parse_args(argv)
    runs, measure_minutes = load_sweep(args.sweep, args.overrides)
    print(describe(runs, measure_minutes, measure=not args.no_measure))
    if args.dry_run:
        return {"done": [], "failed": [], "runs": runs}
    result = run_sweep(runs, measure=not args.no_measure)
    if result["failed"] and argv is None:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
