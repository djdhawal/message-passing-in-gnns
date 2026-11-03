"""Per-subgraph structure summaries and the Gate S0 table.

Every summary is computed on the largest connected component (LCC) of a subgraph:

    n_nodes, n_edges, mean_degree, lcc_size     (whole subgraph; degree = undirected degree)
    eff_diameter        90th percentile of hop distances from up to `n_sources` sampled LCC nodes
    spectral_gap        second-smallest eigenvalue of the normalized Laplacian I - D^-1/2 A D^-1/2
    mean_res            mean effective resistance R_uv over `n_pairs` uniformly sampled LCC pairs
    mean_excess_res     mean of R_uv - (1/d_u + 1/d_v) on the same pairs (von Luxburg et al. 2014:
                        on large well-connected graphs R_uv -> 1/d_u + 1/d_v, so raw R measures degree)
    kirchhoff           Kirchhoff index N * tr(L^+) of the LCC (sum of R over all unordered pairs)
    spearman_res_invdeg Spearman correlation of R_uv with 1/d_u + 1/d_v over the sampled pairs
                        (Gate S0: how much the degree correction matters)

Resistance is exact via L^+ = (L + J/n)^-1 - J/n on the LCC (dense; fine up to ~10k nodes).

    python -m gnn_mech.social.structure twitch-gamers reddit-body --root ./data --out runs/social/s0
"""
from __future__ import annotations

import os
import os.path as osp
from typing import Optional

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr

SUMMARY_KEYS = ("n_nodes", "n_edges", "mean_degree", "lcc_size", "eff_diameter", "spectral_gap", "mean_res",
                "mean_excess_res", "kirchhoff", "spearman_res_invdeg")


def adjacency(edge_index, n: int) -> sp.csr_matrix:
    ei = np.asarray(edge_index)
    keep = ei[0] != ei[1]
    a = sp.csr_matrix((np.ones(keep.sum()), (ei[0][keep], ei[1][keep])), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float64)
    return a.tocsr()


def lcc(a: sp.csr_matrix) -> np.ndarray:
    """Node indices of the largest connected component."""
    if a.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    _, lab = connected_components(a, directed=False)
    return np.flatnonzero(lab == np.bincount(lab).argmax())


def laplacian_pinv(a: sp.csr_matrix) -> np.ndarray:
    """Pseudo-inverse of L = D - A for a *connected* graph: (L + J/n)^-1 - J/n."""
    n = a.shape[0]
    deg = np.asarray(a.sum(1)).ravel()
    L = np.diag(deg) - a.toarray()
    M = L + 1.0 / n
    inv = sla.cho_solve(sla.cho_factor(M, lower=True), np.eye(n))
    return inv - 1.0 / n


def resistance(lp: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = np.diag(lp)
    return d[u] + d[v] - 2.0 * lp[u, v]


def sample_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Distinct-node pairs, all of them if there are few, else `n_pairs` uniform samples."""
    if n * (n - 1) // 2 <= n_pairs:
        u, v = np.triu_indices(n, 1)
        return u, v
    u = rng.integers(0, n, n_pairs)
    v = (u + rng.integers(1, n, n_pairs)) % n
    return u, v


def subgraph_summary(edge_index, n: int, n_pairs: int = 2000, n_sources: int = 64, seed: int = 0,
                     max_exact: int = 10000) -> dict:
    """Structure summary of one subgraph (see module docstring); resistance stats are nan above max_exact."""
    rng = np.random.default_rng(seed)
    a = adjacency(edge_index, n)
    deg_all = np.asarray(a.sum(1)).ravel()
    out = {"n_nodes": float(n), "n_edges": float(a.nnz // 2), "mean_degree": float(deg_all.mean()) if n else np.nan}
    nodes = lcc(a)
    m = nodes.size
    out["lcc_size"] = float(m)
    for k in SUMMARY_KEYS[4:]:
        out[k] = np.nan
    if m < 3:
        return out
    al = a[nodes][:, nodes].tocsr()
    deg = np.asarray(al.sum(1)).ravel()
    src = rng.choice(m, size=min(n_sources, m), replace=False)
    hops = shortest_path(al, method="D", unweighted=True, directed=False, indices=src)
    hops = hops[:, :][np.isfinite(hops) & (hops > 0)]
    out["eff_diameter"] = float(np.percentile(hops, 90))
    dis = 1.0 / np.sqrt(deg)
    norm_l = sp.identity(m) - sp.diags(dis) @ al @ sp.diags(dis)
    if m <= 3000:
        ev = np.linalg.eigvalsh(norm_l.toarray())
        out["spectral_gap"] = float(ev[1])
    else:
        ev = sp.linalg.eigsh(norm_l.tocsc(), k=2, sigma=-1e-3, which="LM", return_eigenvectors=False)
        out["spectral_gap"] = float(np.sort(ev)[1])
    if m > max_exact:
        return out
    lp = laplacian_pinv(al)
    u, v = sample_pairs(m, n_pairs, rng)
    r = resistance(lp, u, v)
    inv = 1.0 / deg[u] + 1.0 / deg[v]
    out["mean_res"] = float(r.mean())
    out["mean_excess_res"] = float((r - inv).mean())
    out["kirchhoff"] = float(m * np.trace(lp))
    out["spearman_res_invdeg"] = float(spearmanr(r, inv).statistic) if np.ptp(inv) > 0 else np.nan
    return out


def summarize_dataset(graphs, name: str = "", split: str = "", max_graphs: Optional[int] = None,
                      **kw) -> list[dict]:
    """One summary row per subgraph `Data` (uses edge_index and num_nodes; graph_id when present)."""
    rows = []
    for i, d in enumerate(graphs):
        if max_graphs is not None and i >= max_graphs:
            break
        gid = int(d.graph_id.view(-1)[0]) if getattr(d, "graph_id", None) is not None else i
        row = {"dataset": name, "split": split, "graph_id": gid}
        row.update(subgraph_summary(d.edge_index.numpy(), int(d.num_nodes), seed=gid, **kw))
        rows.append(row)
    return rows


def structure_table(names, root: str = "./data", max_graphs: Optional[int] = 10, splits=("test",),
                    out_csv: Optional[str] = None, data_overrides: Optional[dict] = None, **kw):
    """Gate S0 table: per-dataset mean (and median) of every summary over subgraphs.

    For each dataset name, builds/loads the subgraphs with the default social `DataConfig` fields
    (plus `data_overrides`), summarizes up to `max_graphs` subgraphs per split, and returns a pandas
    DataFrame with one row per dataset (columns: dataset, n_subgraphs, <key>, <key>_median ...).
    With `out_csv`, writes it there and the per-subgraph rows to `<out_csv stem>_subgraphs.csv`.
    """
    import pandas as pd

    from ..config import DataConfig
    from .dataset import load_social

    if isinstance(names, str):
        names = [names]
    per_graph = []
    for name in names:
        cfg = DataConfig(name=name, root=root, pe_dim=0)
        for k, v in (data_overrides or {}).items():
            setattr(cfg, k, v)
        ds = dict(zip(("train", "val", "test"), load_social(cfg)[:3]))
        for split in splits:
            per_graph += summarize_dataset([ds[split][i] for i in range(len(ds[split]))], name, split,
                                           max_graphs, **kw)
    df = pd.DataFrame(per_graph)
    agg = df.groupby("dataset", sort=False)[list(SUMMARY_KEYS)]
    table = agg.mean()
    med = agg.median().add_suffix("_median")
    table = pd.concat([table, med], axis=1)
    table.insert(0, "n_subgraphs", df.groupby("dataset", sort=False).size())
    table = table.reset_index()
    if out_csv:
        os.makedirs(osp.dirname(osp.abspath(out_csv)), exist_ok=True)
        table.to_csv(out_csv, index=False)
        df.to_csv(osp.splitext(out_csv)[0] + "_subgraphs.csv", index=False)
    return table


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Gate S0 structure table")
    p.add_argument("names", nargs="+")
    p.add_argument("--root", default="./data")
    p.add_argument("--out", default="./runs/social/s0/structure.csv")
    p.add_argument("--max-graphs", type=int, default=10)
    p.add_argument("--subgraph-size", type=int, default=2000)
    args = p.parse_args(argv)
    t = structure_table(args.names, args.root, args.max_graphs, out_csv=args.out,
                        data_overrides={"subgraph_size": args.subgraph_size})
    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t[["dataset", "n_subgraphs"] + list(SUMMARY_KEYS)].round(4).to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
