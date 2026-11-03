"""Training loop, evaluation and resume.

`train` is resumable: it saves `checkpoint.pt` (model, optimizer, scheduler,
best state, history, RNG states) every `cfg.train.checkpoint_every` epochs and
continues from it when called again on the same `RunDir`.

Graph level (`info.task_level == "graph"`, the default): multilabel -> BCE, metric
`ap`; regression -> L1, metric `mae`. Node level (`info.task_level == "node"`, the
social graphs): loss and metrics only over `batch.y_mask` nodes, pooled across the
batch (and, for metrics, across the whole split):

    task         loss             selection metric   also reported
    binary       BCE-with-logits  auroc              ap
    multiclass   cross-entropy    acc
    regression   MSE              r2                 mae

`train.batch_size` counts graphs (subgraphs) per batch.
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
from sklearn.metrics import average_precision_score, roc_auc_score
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


NODE_TASKS = ("binary", "multiclass", "regression")
_NODE_METRIC = {"binary": "auroc", "multiclass": "acc", "regression": "r2"}
HIGHER_IS_BETTER = {"ap": True, "auroc": True, "acc": True, "r2": True, "mae": False}


def task_level(info: Any) -> str:
    """info.task_level, defaulting to "graph" for DataInfo-like objects without it."""
    return getattr(info, "task_level", "graph")


def _metric_name(info: Any) -> str:
    """Model-selection metric: graph level "ap" | "mae"; node level "auroc" | "acc" | "r2"."""
    if task_level(info) == "node":
        if info.task not in _NODE_METRIC:
            raise ValueError(f"node-level task must be one of {NODE_TASKS}, got {info.task!r}")
        return _NODE_METRIC[info.task]
    return "ap" if info.task == "multilabel" else "mae"


def _loss(out: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    y = y.view(out.shape).float()
    if task == "multilabel":
        return F.binary_cross_entropy_with_logits(out, y)
    return F.l1_loss(out, y)


def _node_mask(batch) -> torch.Tensor:
    """Supervised nodes: batch.y_mask, or every node when the batch has no mask."""
    m = getattr(batch, "y_mask", None)
    if m is None:
        return torch.ones(batch.num_nodes, dtype=torch.bool, device=batch.y.device)
    return m.view(-1).bool()


def _node_targets(out: torch.Tensor, y: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(logits, y) of masked nodes in loss form: multiclass -> ([n, C], long [n]); else ([n], float [n])."""
    if task == "multiclass":
        return out, y.view(-1).long()
    return out.reshape(out.shape[0], -1)[:, 0], y.view(-1).float()


def _node_loss(out: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    """Mean loss over the given (already masked) nodes."""
    out, y = _node_targets(out, y, task)
    if task == "binary":
        return F.binary_cross_entropy_with_logits(out, y)
    if task == "multiclass":
        return F.cross_entropy(out, y)
    if task == "regression":
        return F.mse_loss(out, y)
    raise ValueError(f"node-level task must be one of {NODE_TASKS}, got {task!r}")


def node_metrics(out: torch.Tensor, y: torch.Tensor, task: str) -> dict:
    """Metrics over masked nodes. binary: loss, auroc, ap; multiclass: loss, acc;
    regression: loss, r2, mae. auroc / ap are nan when only one class is present,
    r2 is nan when y is constant, and everything is nan for zero nodes."""
    nan = float("nan")
    if out.shape[0] == 0:
        keys = {"binary": ("auroc", "ap"), "multiclass": ("acc",), "regression": ("r2", "mae")}[task]
        return {"loss": nan, **{k: nan for k in keys}}
    loss = _node_loss(out, y, task).item()
    o, t = _node_targets(out, y, task)
    if task == "binary":
        t_np, s_np = t.numpy(), torch.sigmoid(o).numpy()
        two = 0 < t_np.sum() < len(t_np)
        return {"loss": loss, "auroc": float(roc_auc_score(t_np, s_np)) if two else nan,
                "ap": float(average_precision_score(t_np, s_np)) if t_np.sum() > 0 else nan}
    if task == "multiclass":
        return {"loss": loss, "acc": (o.argmax(dim=-1) == t).float().mean().item()}
    ss_res = (o - t).pow(2).sum().item()
    ss_tot = (t - t.mean()).pow(2).sum().item()
    return {"loss": loss, "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else nan,
            "mae": (o - t).abs().mean().item()}


def macro_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Macro average precision over columns with at least one positive (nan if none)."""
    aps = [average_precision_score(y_true[:, c], scores[:, c])
           for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0]
    return float(np.mean(aps)) if aps else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, info: Any, device: torch.device) -> dict:
    """Graph level: {"loss", "ap"} for multilabel tasks, {"loss", "mae"} for regression.
    Node level: `node_metrics` over the y_mask nodes of the whole loader."""
    model.eval()
    outs, ys = [], []
    if task_level(info) == "node":
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).float()
            m = _node_mask(batch)
            outs.append(out[m].cpu())
            ys.append(batch.y.view(-1)[m].cpu())
        return node_metrics(torch.cat(outs), torch.cat(ys), info.task)
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
def predict(model: nn.Module, dataset, batch_size: int, device: torch.device,
            level: str = "graph") -> dict[str, np.ndarray]:
    """Per-graph outputs in dataset order: {"graph_id" [G], "y" [G, out], "logits" [G, out]}.

    `graph_id` falls back to the dataset index for graphs without one. With
    level="node": one entry per y_mask node, in dataset then node order:
    {"graph_id" [n], "node_id" [n], "y" [n], "logits" [n, out]}; `node_id` is
    `batch.node_id` (index in the full graph), or the local node index without it.
    """
    if level == "node":
        return _predict_nodes(model, dataset, batch_size, device)
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


@torch.no_grad()
def _predict_nodes(model: nn.Module, dataset, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    gids, nids, ys, outs = [], [], [], []
    offset = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        out = model(batch).float()
        m = _node_mask(batch)
        b = batch.batch if getattr(batch, "batch", None) is not None else torch.zeros(
            batch.num_nodes, dtype=torch.long, device=out.device)
        gid = getattr(batch, "graph_id", None)
        gid = torch.arange(offset, offset + batch.num_graphs, device=out.device) if gid is None else gid.view(-1)
        nid = getattr(batch, "node_id", None)
        if nid is None:
            nid = torch.arange(batch.num_nodes, device=out.device) - batch.ptr[b] if hasattr(batch, "ptr") \
                else torch.arange(batch.num_nodes, device=out.device)
        gids.append(gid[b][m].cpu())
        nids.append(nid.view(-1)[m].cpu())
        ys.append(batch.y.view(-1)[m].cpu())
        outs.append(out[m].cpu())
        offset += batch.num_graphs
    return {"graph_id": torch.cat(gids).numpy().astype(np.int64),
            "node_id": torch.cat(nids).numpy().astype(np.int64),
            "y": torch.cat(ys).numpy(), "logits": torch.cat(outs).numpy()}


def _train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                 info: Any, device: torch.device, grad_clip: float) -> float:
    """One pass over `loader`; returns the mean training loss, weighted by graphs
    (graph level) or by supervised nodes (node level)."""
    model.train()
    node = task_level(info) == "node"
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        if node:
            m = _node_mask(batch)
            count = int(m.sum())
            if count == 0:
                continue
        opt.zero_grad()
        out = model(batch)
        if node:
            loss = _node_loss(out[m], batch.y.view(-1)[m], info.task)
        else:
            loss = _loss(out, batch.y, info.task)
            count = batch.num_graphs
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += loss.item() * count
        n += count
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
    higher_better = HIGHER_IS_BETTER[metric]

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
