"""Run directories, JSON results and resumable checkpoints.

Layout: <out_dir>/<exp_name>/<config_hash>/seed<seed>/
    config.json        full config
    history.json       per-epoch metrics
    metrics.json       final metrics (best val / test)
    measurements.json  mechanistic measurements
    checkpoint.pt      latest resumable state (model, optimizer, scheduler, epoch)
    best.pt            model weights at best validation score
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return _to_jsonable(obj.detach().cpu().numpy())
    return obj


class RunDir:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = Path(cfg.out_dir) / cfg.exp_name / cfg.hash() / f"seed{cfg.seed}"
        self.path.mkdir(parents=True, exist_ok=True)
        self.save_json("config.json", cfg.to_dict())

    def file(self, name: str) -> Path:
        return self.path / name

    def save_json(self, name: str, obj: Any) -> None:
        tmp = self.file(name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(_to_jsonable(obj), f, indent=2)
        os.replace(tmp, self.file(name))

    def load_json(self, name: str, default: Any = None) -> Any:
        p = self.file(name)
        if not p.exists():
            return default
        with open(p) as f:
            return json.load(f)

    def save_torch(self, name: str, obj: Any) -> None:
        tmp = self.file(name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, self.file(name))

    def load_torch(self, name: str, map_location: Any = "cpu") -> Any:
        p = self.file(name)
        if not p.exists():
            return None
        return torch.load(p, map_location=map_location, weights_only=False)

    def is_complete(self) -> bool:
        return self.file("metrics.json").exists()
