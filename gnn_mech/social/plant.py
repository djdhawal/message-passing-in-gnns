"""Planted semi-synthetic targets with a known range k on real social graphs (docs/social_plan.md, S3).

The real graph and the real node features are kept (the features become nuisance inputs); `p` planted
input channels Z are appended, and the target y is an exact function of Z on the graph, so its range is
known. Target names (`data.target`): ``plant:<family>:<k>[:s<seed>]``, e.g. ``plant:retrieve:3:s1``.

Everything is computed on ``graph.edge_index`` treated as undirected (the follow layer for Higgs), i.e. on
the *full* graph, so the target is the same whatever sampler later cuts the graph. Hop distances are
exact (blocked BFS by sparse-dense products, ``_levels``). Nodes whose target is undefined get NaN
(-> ``y_mask`` False in `dataset.py`).

Families
--------
``shell_mean`` (sanity). One channel z_u ~ N(0, 1); y_v = mean of z_u over S_k(v) = {u : d(v, u) = k}.
    Undefined when S_k(v) is empty. Jacobian: dy_v/dz_u = 1/|S_k(v)| on the shell, 0 elsewhere.
    Caveat: when S_k(v) is almost "everything at distance >= k" (k >= 3 on Higgs), the sum over the shell
    is (total - sum over the (k-1)-ball - sum beyond k), and the total is a constant of the dataset, so a
    model can fit y from the *inner* ball. The Jacobian of the target is still exactly k-range, but the
    task stops being k-range in practice; `make` reports this as ``oracle["inner_leak_r2"]`` (R2 of y on
    the inner-ball sum and 1/|S|) and `shell_stats` as ``shell_of_rest``.

``retrieve`` (the main test). Channels: key code (``key_dim`` dims), query code (``key_dim``), value (1).
    Every node u has a discrete key c_u ~ Uniform{0..K-1}, shown as the code vector C[c_u] (C: K random
    unit-norm Gaussian codes, so attention can match keys by dot product), and a value w_u ~ N(0, 1).
    Node v's query q_v is chosen as follows:
      1. candidates = shell nodes u in S_k(v) whose key does not occur anywhere in the inner ball
         B_{k-1}(v) = {u : d(v, u) <= k-1} (v itself included);
      2. u* = a uniformly random candidate (seeded); q_v = c_{u*}; v's query channel = C[q_v];
      3. y_v = mean of w_u over the matches M(v) = {u in S_k(v) : c_u = q_v} (u* and any other shell node
         that happens to share the key). Undefined (NaN, query channel 0) if there is no candidate.
    Why this is k-range: by construction no node closer than k carries the query key, so nothing inside
    the (k-1)-ball matches; y depends on the values of the matching shell nodes only (dy_v/dw_u = 1/|M(v)|
    for u in M(v), 0 elsewhere; keys and queries are discrete, so their partial derivatives are 0), and
    the values are i.i.d. noise, so no closer node carries information about them. Nodes beyond k may
    share the key: they are the distractors a k-hop-unaware model (e.g. global attention) must reject.
    Why the expected number of matches stays ~1 at every k: social shells grow very fast with k (Higgs:
    median |S_1| ~ 30, |S_2| in the thousands, k = 3 covers most of the graph), so a fixed key alphabet
    would give either huge match counts at large k or trivially unique keys at small k. The number of key
    values is therefore set per k from the ball sizes: K_k = ceil(quantile(|B_k(v)|, key_quantile)) over
    eligible nodes (default 0.9, clipped to [2, N]). The query key is drawn from the shell itself, so there
    is always at least one match, and the expected number of *extra* matches is (|S_k(v)| - 1)/K_k <= 1
    for ~90% of nodes; the chance that a random shell key also appears in the inner ball,
    ~|B_{k-1}|/K_k, is small, so few nodes lose their target in step 1. `make` reports the realised match
    counts (``meta["matches"]``).

``analytic_range`` computes the normalized Bamberger range (as `gnn_mech.measure.range`) of the target
function itself from its exact Jacobian rows, with hop distances from an independent scipy BFS; it is
k for both families by construction (and is tested to be so). Resistance range is added when the graph's
largest component has at most `MAX_RES_NODES` nodes (exact resistance matrix).
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp

FAMILIES = ("shell_mean", "retrieve")
KEY_DIM = 16                 # dimensions of the retrieve key / query codes
KEY_QUANTILE = 0.9           # K_k = this quantile of the k-ball sizes (see module docstring)
BLOCK_ELEMENTS = 1 << 23     # source block size: B * N <= this. Never an N x N matrix: the BFS runs on
                             # [N, B] frontier blocks (~8 MB per bool block; peak ~ (k + 8) blocks, < 0.5 GB)
MAX_RES_NODES = 5000         # exact resistance range only on graphs (LCC) up to this size
_PATTERN = re.compile(r"^plant:([a-z_]+):(\d+)(?::s(\d+))?$")
_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_CACHE_SIZE = 16


# ------------------------------------------------------------------------ names


def is_plant(target) -> bool:
    """True for target names of the form ``plant:...`` (valid or not; `parse` validates)."""
    return isinstance(target, str) and target.startswith("plant:")


def parse(target: str) -> tuple[str, int, int]:
    """``"plant:<family>:<k>[:s<seed>]"`` -> (family, k, seed); seed defaults to 0."""
    m = _PATTERN.match(target or "")
    if not m:
        raise ValueError(f"bad planted target {target!r}; expected 'plant:<family>:<k>[:s<seed>]'")
    family, k, seed = m.group(1), int(m.group(2)), int(m.group(3) or 0)
    if family not in FAMILIES:
        raise ValueError(f"unknown planted family {family!r}; expected one of {FAMILIES}")
    if k < 1:
        raise ValueError(f"planted range k must be >= 1, got {k}")
    return family, k, seed


def name(family: str, k: int, seed: int = 0) -> str:
    """Inverse of `parse` (the seed suffix is omitted for seed 0)."""
    return f"plant:{family}:{int(k)}" + (f":s{int(seed)}" if seed else "")


def channel_names(family: str, key_dim: int = KEY_DIM) -> list[str]:
    if family == "shell_mean":
        return ["plant_z0"]
    if family == "retrieve":
        return ([f"plant_key{i}" for i in range(key_dim)] + [f"plant_query{i}" for i in range(key_dim)]
                + ["plant_value"])
    raise ValueError(f"unknown planted family {family!r}")


# ---------------------------------------------------------------- graph helpers


def _edges(graph) -> np.ndarray:
    ei = graph.edge_index
    ei = ei.detach().cpu().numpy() if hasattr(ei, "detach") else np.asarray(ei)
    return np.asarray(ei, dtype=np.int64).reshape(2, -1)


def adjacency(graph) -> sp.csr_matrix:
    """Symmetric 0/1 float32 CSR adjacency of `graph.edge_index` (undirected, no self loops)."""
    n = int(graph.num_nodes)
    ei = _edges(graph)
    keep = ei[0] != ei[1]
    r, c = ei[0][keep], ei[1][keep]
    a = sp.coo_matrix((np.ones(2 * r.size, np.float32), (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr()
    a.sum_duplicates()
    a.data[:] = 1.0
    return a


def fingerprint(graph) -> str:
    """Hash of (num_nodes, edge_index): the cache key of everything computed on the graph."""
    h = hashlib.sha1(str(int(graph.num_nodes)).encode())
    h.update(np.ascontiguousarray(_edges(graph)).tobytes())
    return h.hexdigest()[:16]


def _cached(key: tuple, fn):
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    val = fn()
    _CACHE[key] = val
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return val


def clear_cache() -> None:
    _CACHE.clear()


def _blocks(n: int, budget: Optional[int] = None):
    budget = BLOCK_ELEMENTS if budget is None else budget
    b = int(np.clip(budget // max(n, 1), 16, max(n, 16)))
    for s in range(0, n, b):
        yield np.arange(s, min(s + b, n))


def _levels(a: sp.csr_matrix, src: np.ndarray, kmax: int) -> list[np.ndarray]:
    """Exact BFS shells for a block of sources: list over d = 0..kmax of bool [N, B] (column j = shell
    {u : d(src[j], u) = d}). Each hop is one sparse-dense product A @ frontier."""
    n = a.shape[0]
    front = np.zeros((n, src.size), dtype=bool)
    front[src, np.arange(src.size)] = True
    seen = front.copy()
    out = [front]
    for d in range(kmax):
        # hop 1 is the sources' adjacency rows (A is symmetric); later hops one sparse-dense product
        nxt = (a[src].toarray().T > 0) if d == 0 else ((a @ front.astype(np.float32)) > 0)
        nxt &= ~seen
        seen |= nxt
        out.append(nxt)
        front = nxt
    return out


def _level_sizes(graph, kmax: int) -> np.ndarray:
    """[kmax + 1, N] int64: |S_d(v)| for d = 0..kmax (cached per graph; reuses a deeper cached run)."""
    fp = fingerprint(graph)
    for key, val in list(_CACHE.items()):
        if key[0] == "sizes" and key[1] == fp and key[2] >= kmax:
            return val[: kmax + 1]

    def run():
        a = adjacency(graph)
        n = a.shape[0]
        sizes = np.zeros((kmax + 1, n), dtype=np.int64)
        for src in _blocks(n):
            for d, lev in enumerate(_levels(a, src, kmax)):
                sizes[d, src] = lev.sum(0)
        return sizes

    return _cached(("sizes", fp, kmax), run)


def _components(graph) -> np.ndarray:
    from scipy.sparse.csgraph import connected_components
    return _cached(("comp", fingerprint(graph)),
                   lambda: connected_components(adjacency(graph), directed=False)[1])


# ------------------------------------------------------------------- statistics


def _dist(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "max": None}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.quantile(x, 0.1)),
            "p90": float(np.quantile(x, 0.9)), "max": float(x.max())}


def shell_stats(graph, ks: Sequence[int] = (1, 2, 3, 4)) -> list[dict]:
    """Per k: shell sizes |S_k(v)|, eligible nodes (non-empty shell), ball coverage and `shell_of_rest`.

    `ball_frac` is |B_k(v)| / N; `shell_of_rest` is |S_k(v)| / |{u reachable : d(v, u) >= k}| averaged over
    eligible nodes: near 1 means the shell is everything outside the (k-1)-ball, so shell sums are a
    constant minus inner-ball sums (see the `shell_mean` caveat in the module docstring).
    """
    ks = sorted({int(k) for k in ks})
    t0 = time.time()
    sizes = _level_sizes(graph, max(ks))
    n = sizes.shape[1]
    comp = _components(graph)
    comp_size = np.bincount(comp)[comp]
    ball = np.cumsum(sizes, axis=0)
    rows = []
    for k in ks:
        s = sizes[k]
        ok = s > 0
        rest = comp_size - ball[k - 1]
        rows.append({"k": k, "n_nodes": int(n), "eligible": int(ok.sum()), "eligible_frac": float(ok.mean()),
                     "shell": _dist(s[ok]), "ball_frac": _dist(ball[k][ok] / n),
                     "shell_of_rest": float(np.mean(s[ok] / np.maximum(rest[ok], 1))) if ok.any() else None})
    rows[0]["seconds"] = round(time.time() - t0, 2) if rows else None
    return rows


def _r2(y: np.ndarray, pred: np.ndarray) -> Optional[float]:
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 3 or np.var(y[ok]) == 0:
        return None
    return float(1 - np.mean((y[ok] - pred[ok]) ** 2) / np.var(y[ok]))


def _ols_r2(y: np.ndarray, X: np.ndarray) -> Optional[float]:
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    if ok.sum() <= X.shape[1] + 2:
        return None
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    beta, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return _r2(y[ok], A @ beta)


# ---------------------------------------------------------------------- targets


def _rng(family: str, k: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), FAMILIES.index(family), int(k), 20261103])


def make(graph, family: str, k: int, seed: int = 0, key_dim: int = KEY_DIM,
         key_quantile: float = KEY_QUANTILE) -> tuple[np.ndarray, np.ndarray, dict]:
    """(Z [N, p] float64 planted input channels, y [N] float64 with NaN where undefined, meta).

    Deterministic in (graph, family, k, seed, key_dim, key_quantile) and cached in memory. `meta` holds
    the specification (family, k, seed, channel names, K, ...), timing, eligible counts, match counts
    (retrieve) and oracle R2s (see module docstring); for retrieve also `keys`, `query` ([N], -1 where
    undefined) and `values`, which `jacobian` / `analytic_range` use.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown planted family {family!r}; expected one of {FAMILIES}")
    if k < 1:
        raise ValueError("k must be >= 1")
    key = ("make", fingerprint(graph), family, int(k), int(seed), int(key_dim), float(key_quantile))
    return _cached(key, lambda: (_make_shell_mean if family == "shell_mean" else _make_retrieve)(
        graph, int(k), int(seed), int(key_dim), float(key_quantile)))


def _make_shell_mean(graph, k, seed, key_dim, key_quantile):
    t0 = time.time()
    a = adjacency(graph)
    n = a.shape[0]
    z = _rng("shell_mean", k, seed).standard_normal(n)
    y = np.full(n, np.nan)
    size = np.zeros(n, dtype=np.int64)
    inner = np.zeros(n)
    for src in _blocks(n):
        lev = _levels(a, src, k)
        shell = lev[k]
        size[src] = shell.sum(0)
        s = z @ shell
        with np.errstate(invalid="ignore", divide="ignore"):
            y[src] = np.where(size[src] > 0, s / np.maximum(size[src], 1), np.nan)
        inner[src] = z @ np.logical_or.reduce(lev[:k])
    ok = size > 0
    inv = np.where(ok, 1.0 / np.maximum(size, 1), np.nan)
    meta = {"family": "shell_mean", "k": k, "seed": seed, "names": channel_names("shell_mean"),
            "n_nodes": int(n), "eligible": int(ok.sum()), "eligible_frac": float(ok.mean()),
            "shell": _dist(size[ok]),
            # y = (const - inner-ball sum - beyond-k sum) / |S|: how much of y the inner ball explains
            "oracle": {"shell_aggregate_r2": _r2(y, y),
                       "inner_leak_r2": _ols_r2(y, np.column_stack([inv, inner * inv]))},
            "seconds": round(time.time() - t0, 2)}
    return z[:, None].copy(), y, meta


def _make_retrieve(graph, k, seed, key_dim, key_quantile):
    t0 = time.time()
    a = adjacency(graph)
    n = a.shape[0]
    sizes = _level_sizes(graph, k)
    ball = sizes.sum(0)
    elig = sizes[k] > 0
    n_keys = int(np.clip(np.ceil(np.quantile(ball[elig], key_quantile)) if elig.any() else 2, 2, max(n, 2)))
    rng = _rng("retrieve", k, seed)
    keys = rng.integers(0, n_keys, size=n)
    values = rng.standard_normal(n)
    codes = rng.standard_normal((n_keys, key_dim))
    codes /= np.linalg.norm(codes, axis=1, keepdims=True)
    prio = rng.random(n)              # candidate choice: argmax of (prio_u + offset_v) mod 1 per v
    offset = rng.random(n)            # (uniform per v, independent of the block layout)
    onehot_t = sp.csr_matrix((np.ones(n, np.float32), (keys, np.arange(n))), shape=(n_keys, n))

    query = np.full(n, -1, dtype=np.int64)
    y = np.full(n, np.nan)
    n_match = np.zeros(n, dtype=np.int64)
    n_cand = np.zeros(n, dtype=np.int64)
    shell_val = np.full(n, np.nan)
    for src in _blocks(n):
        lev = _levels(a, src, k)
        shell = lev[k]
        inner = np.logical_or.reduce(lev[:k])
        present = (onehot_t @ inner.astype(np.float32)) > 0          # [K, B]: key occurs in the inner ball
        cand = shell & ~present[keys]                                # [N, B]
        n_cand[src] = cand.sum(0)
        pri = np.where(cand, ((prio[:, None] + offset[None, src]) % 1.0).astype(np.float32), np.float32(-1.0))
        ustar = pri.argmax(0)
        ok = n_cand[src] > 0
        q = np.where(ok, keys[ustar], -1)
        match = shell & (keys[:, None] == q[None, :])
        m = match.sum(0)
        query[src] = q
        n_match[src] = np.where(ok, m, 0)
        y[src] = np.where(ok, (values @ match) / np.maximum(m, 1), np.nan)
        s = shell.sum(0)
        shell_val[src] = np.where(s > 0, (values @ shell) / np.maximum(s, 1), np.nan)
    ok = query >= 0
    qcode = np.zeros((n, key_dim))
    qcode[ok] = codes[query[ok]]
    Z = np.concatenate([codes[keys], qcode, values[:, None]], axis=1)
    # global key match: what a hop-unaware retriever gets (all nodes with the query key, any distance)
    gsum = np.bincount(keys, weights=values, minlength=n_keys)
    gcnt = np.bincount(keys, minlength=n_keys)
    glob = np.where(ok, gsum[np.maximum(query, 0)] / np.maximum(gcnt[np.maximum(query, 0)], 1), np.nan)
    mm = n_match[ok]
    meta = {"family": "retrieve", "k": k, "seed": seed, "names": channel_names("retrieve", key_dim),
            "key_dim": key_dim, "key_quantile": key_quantile, "n_keys": n_keys, "n_nodes": int(n),
            "eligible": int(ok.sum()), "eligible_frac": float(ok.mean()),
            "shell_nonempty": int(elig.sum()), "lost_to_inner_keys": int((elig & ~ok).sum()),
            "shell": _dist(sizes[k][ok]),
            "matches": {**_dist(mm), "frac_unique": float(np.mean(mm == 1)) if mm.size else None},
            "global_matches": _dist(gcnt[query[ok]]),
            "oracle": {"key_match_in_shell_r2": _r2(y, y), "shell_value_mean_r2": _r2(y, shell_val),
                       "global_key_match_r2": _r2(y, glob)},
            "keys": keys, "query": query, "values": values, "seconds": round(time.time() - t0, 2)}
    return Z, y, meta


# --------------------------------------------------------------- exact Jacobian


def _hops_from(graph, nodes: np.ndarray) -> np.ndarray:
    """[len(nodes), N] hop distances (inf if unreachable) by scipy BFS, independent of `_levels`."""
    from scipy.sparse.csgraph import shortest_path
    return shortest_path(adjacency(graph), method="D", unweighted=True, directed=False,
                         indices=np.asarray(nodes, dtype=np.int64))


def jacobian(graph, family: str, k: int, seed: int = 0, v: int = 0, hops: Optional[np.ndarray] = None,
             **kw) -> np.ndarray:
    """Exact Jacobian dy_v / dZ [N, p] of the planted target of node v w.r.t. the planted channels.

    shell_mean: 1/|S_k(v)| on the shell. retrieve: 1/|M(v)| in the value channel of the matches and 0 in
    the key / query channels (they are discrete, so y is piecewise constant in them). All zero when y_v is
    undefined. `hops`: v's hop-distance row, if already computed.
    """
    Z, y, meta = make(graph, family, k, seed, **kw)
    J = np.zeros(Z.shape)
    if not np.isfinite(y[v]):
        return J
    if hops is None:
        hops = _hops_from(graph, [v])[0]
    shell = hops == k
    if family == "shell_mean":
        J[shell, 0] = 1.0 / shell.sum()
    else:
        match = shell & (meta["keys"] == meta["query"][v])
        J[match, -1] = 1.0 / match.sum()
    return J


def analytic_range(graph, family: str, k: int, seed: int = 0, nodes: Optional[Sequence[int]] = None,
                   n_sample: int = 256, resistance="auto", **kw) -> dict:
    """Normalized Bamberger range of the planted target from its exact Jacobian (I_vu = sum_c |J_vuc|).

    `nodes`: target nodes (default: `n_sample` random eligible nodes, seeded). Returns {"family", "k",
    "n", "range_hops" (mean), "range_hops_min", "range_hops_max", "range_res" (mean or None),
    "per_node": [{"v", "range_hops", "range_res"}]}. Hop range equals k exactly by construction.
    `resistance`: True / False / "auto" (only when the largest component has <= MAX_RES_NODES nodes).
    """
    from ..measure.range import normalized_range

    Z, y, meta = make(graph, family, k, seed, **kw)
    if nodes is None:
        elig = np.flatnonzero(np.isfinite(y))
        rng = np.random.default_rng([int(seed), int(k), 7])
        nodes = np.sort(rng.choice(elig, min(n_sample, elig.size), replace=False)) if elig.size else elig
    nodes = np.asarray(nodes, dtype=np.int64)
    nodes = nodes[np.isfinite(y[nodes])]
    rows = []
    res = None
    if nodes.size:
        hops = _hops_from(graph, nodes)
        do_res = resistance is True
        if resistance == "auto" or do_res:
            from ..measure.resistance import effective_resistance_matrix, lcc_nodes
            lcc = np.asarray(lcc_nodes(graph.edge_index, int(graph.num_nodes)))
            do_res = do_res or lcc.size <= MAX_RES_NODES
            if do_res:
                R = effective_resistance_matrix(graph.edge_index, int(graph.num_nodes), nodes=lcc)
                pos = {int(u): i for i, u in enumerate(lcc)}
        for i, v in enumerate(nodes):
            infl = np.abs(jacobian(graph, family, k, seed, int(v), hops=hops[i], **kw)).sum(1)   # I_vu
            row = {"v": int(v), "range_hops": normalized_range(infl, hops[i]), "range_res": None}
            if do_res and int(v) in pos:
                d = np.full(infl.size, np.inf)
                d[lcc] = R[pos[int(v)]]
                row["range_res"] = normalized_range(infl, d)
            rows.append(row)
    h = np.array([r["range_hops"] for r in rows if r["range_hops"] is not None], dtype=float)
    rr = np.array([r["range_res"] for r in rows if r["range_res"] is not None], dtype=float)
    return {"family": family, "k": int(k), "seed": int(seed), "n": int(h.size),
            "range_hops": float(h.mean()) if h.size else None,
            "range_hops_min": float(h.min()) if h.size else None,
            "range_hops_max": float(h.max()) if h.size else None,
            "range_res": float(rr.mean()) if rr.size else None, "per_node": rows}


def spec(meta: dict) -> dict:
    """JSON-safe planted-target specification (meta without the per-node arrays)."""
    return {k: v for k, v in meta.items() if not isinstance(v, np.ndarray)}
