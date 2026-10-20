"""Choose the Reddit regression target from the data (Phase 2, B2).

The pilot target (feature 0, character count) was nearly a function of the node's
own inputs (near-duplicates such as "characters without spaces" stayed in x), so
it could not reward multi-hop information. For each candidate column:

1. Inputs are every other column, minus those with |r| > 0.9 to the target
   (Pearson on source nodes).
2. On a *random* split of the source nodes, ridge regression (RidgeCV on
   standardized features) is fit on
       own        the node's own inputs                         R2(a)
       hop1..hopK own + propagated means for 1..k hops (SGC)   R2(k)
   plus a small-MLP check on own and own + K hops.
3. neighbour_gain = R2(K) - R2(own); multi_hop_gain = max_{k>=2} R2(k) - R2(1).
   Skewed non-negative targets (skew > 1) are modelled as log1p(y).

Propagation: only source nodes have features, so hop k averages the hop-(k-1)
features of the neighbours that have them (non-source nodes get features from
hop 1 on, and pass them along); nodes with no such neighbour get 0. Edges are
taken undirected.

    python -m gnn_mech.reddit.target_select [--root data/reddit] [--cached-pt PT]
                                            [--out runs/reddit/target_selection.json] [--synthetic]
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Optional

import numpy as np
from scipy.stats import skew
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_NAMES, LIWC_COLS, SENTIMENT_COLS, undirected_adjacency

DEFAULT_CANDIDATES = SENTIMENT_COLS + LIWC_COLS


def drop_correlated(X: np.ndarray, y: np.ndarray, thresh: float = 0.9) -> np.ndarray:
    """Indices of columns of X with |corr(X[:, j], y)| <= thresh (constant columns dropped too)."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(0) / denom
    return np.flatnonzero(np.isfinite(r) & (np.abs(r) <= thresh))


def propagate(X: np.ndarray, edge_index, valid: np.ndarray, k: int) -> list[np.ndarray]:
    """[H_1, ..., H_k]: H_j[v] = mean of H_{j-1}[u] over neighbours u with valid_{j-1}[u]."""
    n = X.shape[0]
    a = undirected_adjacency(edge_index, n)
    H, ok = np.where(valid[:, None], X, 0.0), valid.astype(float)
    out = []
    for _ in range(k):
        cnt = a @ ok
        H = (a @ (H * ok[:, None])) / np.maximum(cnt, 1)[:, None]
        ok = (cnt > 0).astype(float)
        out.append(H)
    return out


def _ridge_r2(Xtr, ytr, Xte, yte) -> float:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    model.fit(Xtr, ytr)
    return float(r2_score(yte, model.predict(Xte)))


def _mlp_r2(Xtr, ytr, Xte, yte, seed: int) -> float:
    model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300,
                                                         early_stopping=True, random_state=seed))
    with warnings.catch_warnings():       # a rough check; non-convergence warnings are noise here
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(Xtr, (ytr - ytr.mean()) / (ytr.std() + 1e-12))
    pred = model.predict(Xte) * (ytr.std() + 1e-12) + ytr.mean()
    return float(r2_score(yte, pred))


def score_target(x: np.ndarray, edge_index, is_source: np.ndarray, target: int, k_max: int = 4,
                 seed: int = 0, test_frac: float = 0.3, corr_thresh: float = 0.9, mlp: bool = True) -> dict:
    """Neighbour and multi-hop gains for predicting column `target` (see module docstring)."""
    src = np.flatnonzero(is_source)
    y_all = x[:, target].astype(float)
    y = y_all[src]
    log = bool(y.min() >= 0 and skew(y) > 1)
    if log:
        y = np.log1p(y)
    others = np.array([j for j in range(x.shape[1]) if j != target])
    keep = others[drop_correlated(x[src][:, others], y_all[src], corr_thresh)]
    X = x[:, keep].astype(float)
    hops = propagate(X, edge_index, np.asarray(is_source, dtype=bool), k_max)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(src.size)
    n_te = int(round(test_frac * src.size))
    te, tr = perm[:n_te], perm[n_te:]

    def feats(k: int) -> np.ndarray:
        return np.hstack([X[src]] + [h[src] for h in hops[:k]])

    r2 = {"own": _ridge_r2(X[src][tr], y[tr], X[src][te], y[te])}
    for k in range(1, k_max + 1):
        F = feats(k)
        r2[f"hop{k}"] = _ridge_r2(F[tr], y[tr], F[te], y[te])
    out = {
        "target": int(target), "name": FEATURE_NAMES[target] if x.shape[1] == len(FEATURE_NAMES) else str(target),
        "log_target": log, "n_inputs": int(keep.size),
        "dropped_correlated": [int(j) for j in others if j not in set(keep)],
        "r2": r2, "neighbour_gain": r2[f"hop{k_max}"] - r2["own"],
        "multi_hop_gain": (max(r2[f"hop{k}"] for k in range(2, k_max + 1)) - r2["hop1"]) if k_max >= 2 else 0.0,
    }
    if mlp:
        Fk = feats(k_max)
        out["mlp_r2"] = {"own": _mlp_r2(X[src][tr], y[tr], X[src][te], y[te], seed),
                         f"hop{k_max}": _mlp_r2(Fk[tr], y[tr], Fk[te], y[te], seed)}
    return out


def rank_targets(data, candidates: Optional[list[int]] = None, **kw) -> list[dict]:
    """score_target for every candidate column, sorted by multi_hop_gain (then neighbour_gain)."""
    x = data.x.numpy()
    is_source = data.is_source.numpy()
    if candidates is None:
        candidates = DEFAULT_CANDIDATES if x.shape[1] == len(FEATURE_NAMES) else list(range(x.shape[1]))
    rows = [score_target(x, data.edge_index.numpy(), is_source, c, **kw) for c in candidates]
    return sorted(rows, key=lambda r: (r["multi_hop_gain"], r["neighbour_gain"]), reverse=True)


def select_target(table: list[dict], min_gain: float = 0.01, own_r2: tuple[float, float] = (0.05, 0.7)) -> dict:
    """Best-ranked target with multi_hop_gain >= min_gain and moderate own-feature R2.

    Returns {"target": row or None, "reason": str}. None means no candidate shows a
    multi-hop gain; then Experiment 1 is reframed as generalization to distant nodes.
    """
    for row in table:
        if row["multi_hop_gain"] >= min_gain and own_r2[0] <= row["r2"]["own"] <= own_r2[1]:
            return {"target": row, "reason": f"multi_hop_gain {row['multi_hop_gain']:.3f} >= {min_gain}, "
                                             f"own R2 {row['r2']['own']:.3f} in {list(own_r2)}"}
    return {"target": None, "reason": f"no candidate with multi_hop_gain >= {min_gain} and own R2 in {list(own_r2)}"}


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="./data/reddit")
    parser.add_argument("--cached-pt", default=None, help="pilot reddit_base_graph.pt (e.g. on Drive)")
    parser.add_argument("--out", default="./runs/reddit/target_selection.json")
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--candidates", type=int, nargs="*", default=None)
    parser.add_argument("--synthetic", action="store_true", help="tiny synthetic graph (smoke test)")
    args = parser.parse_args(argv)
    if args.synthetic:
        from .data import synthetic_graph
        data = synthetic_graph()
    else:
        from .data import load_reddit_graph
        data = load_reddit_graph(args.root, args.cached_pt)
    table = rank_targets(data, args.candidates, k_max=args.k_max, seed=args.seed, mlp=not args.no_mlp)
    choice = select_target(table)
    result = {"selected": choice["target"]["target"] if choice["target"] else None, "reason": choice["reason"],
              "k_max": args.k_max, "seed": args.seed, "table": table}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    for r in table[:10]:
        print(f"{r['target']:3d} {r['name']:<22s} own R2 {r['r2']['own']:+.3f}  "
              f"neighbour gain {r['neighbour_gain']:+.3f}  multi-hop gain {r['multi_hop_gain']:+.3f}")
    print(f"selected: {result['selected']} ({choice['reason']}) -> {args.out}")
    return result


if __name__ == "__main__":
    main()
