"""Training loop, evaluation and resume.

`train` is resumable: it saves `checkpoint.pt` (model, optimizer, scheduler,
best state, history, RNG states) every `cfg.train.checkpoint_every` epochs and
continues from it when called again on the same `RunDir`.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader

from .config import Config
from .results import RunDir

LOG_EVERY = 10


def resolve_device(name: str) -> torch.device:
    """Map "auto" | "cpu" | "cuda" (or "cuda:N") to a torch.device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_cosine(epoch: int, warmup_epochs: int, epochs: int) -> float:
    """LR multiplier for 0-based `epoch`: linear warmup, then cosine decay to 0 at `epochs`."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    decay = max(1, epochs - warmup_epochs)
    progress = min(1.0, (epoch - warmup_epochs) / decay)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _metric_name(info: Any) -> str:
    return "ap" if info.task == "multilabel" else "mae"


def _loss(out: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    y = y.view(out.shape).float()
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(out, y)
    return F.l1_loss(out, y)


def macro_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Macro average precision over columns with at least one positive (nan if none)."""
    aps = [average_precision_score(y_true[:, c], scores[:, c])
           for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0]
    return float(np.mean(aps)) if aps else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, info: Any, device: torch.device) -> dict:
    """{"loss", "ap"} for multilabel tasks, {"loss", "mae"} for regression."""
    model.eval()
    outs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        outs.append(out.float().cpu())
        ys.append(batch.y.view(out.shape).float().cpu())
    out, y = torch.cat(outs), torch.cat(ys)
    loss = _loss(out, y, info.task).item()
    if info.task == "multilabel":
        return {"loss": loss, "ap": macro_ap(y.numpy(), torch.sigmoid(out).numpy())}
    return {"loss": loss, "mae": (out - y).abs().mean().item()}


@torch.no_grad()
def predict(model: nn.Module, dataset, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    """Per-graph outputs in dataset order: {"graph_id" [G], "y" [G, out], "logits" [G, out]}.

    `graph_id` falls back to the dataset index for graphs without one.
    """
    model.eval()
    gids, ys, outs = [], [], []
    offset = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        out = model(batch).float()
        gid = getattr(batch, "graph_id", None)
        if gid is None:
            gid = torch.arange(offset, offset + batch.num_graphs)
        gids.append(gid.reshape(-1).cpu())
        ys.append(batch.y.view(out.shape).float().cpu())
        outs.append(out.cpu())
        offset += batch.num_graphs
    return {"graph_id": torch.cat(gids).numpy().astype(np.int64),
            "y": torch.cat(ys).numpy(), "logits": torch.cat(outs).numpy()}


def _train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                 info: Any, device: torch.device, grad_clip: float) -> float:
    """One pass over `loader`; returns the graph-weighted mean training loss."""
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        loss = _loss(model(batch), batch.y, info.task)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


def _rng_state(gen: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": gen.get_state(),
    }


def _set_rng_state(state: dict, gen: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    gen.set_state(state["loader"])


def _cpu_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train(cfg: Config, model: nn.Module, train_ds, val_ds, test_ds, info: Any, run: RunDir) -> dict:
    """Train with AdamW + warmup/cosine, keep the best-val model, write metrics.json."""
    tc = cfg.train
    device = resolve_device(tc.device)
    model.to(device)
    metric = _metric_name(info)
    higher_better = metric == "ap"

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    nw = cfg.data.num_workers
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True,
                              num_workers=nw, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=tc.batch_size, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=tc.batch_size, num_workers=nw)

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: warmup_cosine(e, tc.warmup_epochs, tc.epochs))

    start_epoch, history, elapsed, stopped = 0, [], 0.0, False
    best: dict = {"epoch": None, "score": None, "val": {}, "test": {}, "bad": 0, "state": None}

    ckpt = run.load_torch("checkpoint.pt", map_location="cpu")
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch, history = ckpt["epoch"], ckpt["history"]
        best, elapsed, stopped = ckpt["best"], ckpt["seconds"], ckpt.get("stopped", False)
        _set_rng_state(ckpt["rng"], gen)
        run.save_json("history.json", history)
        if best["state"] is not None:
            run.save_torch("best.pt", best["state"])
        print(f"[resume] {run.path} from epoch {start_epoch}")

    def save_checkpoint(epoch: int) -> None:
        run.save_torch("checkpoint.pt", {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "epoch": epoch, "history": history,
            "best": best, "seconds": elapsed, "stopped": stopped, "rng": _rng_state(gen),
        })

    last_epoch = start_epoch if stopped else tc.epochs
    for epoch in range(start_epoch + 1, last_epoch + 1):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        train_loss = _train_epoch(model, train_loader, opt, info, device, tc.grad_clip)
        sched.step()
        val = evaluate(model, val_loader, info, device)

        score = val[metric]
        improved = not math.isnan(score) and (
            best["score"] is None or (score > best["score"] if higher_better else score < best["score"]))
        if improved:
            best.update(epoch=epoch, score=score, val=val, bad=0,
                        test=evaluate(model, test_loader, info, device), state=_cpu_state(model))
            run.save_torch("best.pt", best["state"])
        else:
            best["bad"] += 1
        stopped = tc.patience > 0 and best["bad"] >= tc.patience

        seconds = time.time() - t0
        elapsed += seconds
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        **{f"val_{k}": v for k, v in val.items()}, "seconds": seconds})
        run.save_json("history.json", history)

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == tc.epochs or stopped:
            best_score = best["score"] if best["score"] is not None else float("nan")
            print(f"ep {epoch:4d} | lr {lr:.2e} | loss {train_loss:.4f} | val {metric} {score:.4f} "
                  f"| best {best_score:.4f}@{best['epoch']} "
                  f"| test@best {best['test'].get(metric, float('nan')):.4f} | {seconds:.1f}s")
        if stopped:
            print(f"early stop at epoch {epoch}")
        if epoch % tc.checkpoint_every == 0 or epoch == tc.epochs or stopped:
            save_checkpoint(epoch)
        if stopped:
            break

    epochs_run = history[-1]["epoch"] if history else 0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics = {
        "best_epoch": best["epoch"],
        "val": best["val"],
        "test": best["test"],
        "n_params": n_params,
        "epochs_run": epochs_run,
        "seconds": elapsed,
        "config_hash": cfg.hash(),
    }
    run.save_json("metrics.json", metrics)
    return metrics
