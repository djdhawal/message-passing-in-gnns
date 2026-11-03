# Social-graph notebooks (Colab)

Each notebook stands alone: it mounts Drive, clones the repo at `BRANCH` (default `main`, set in the
first code cell), installs the package, and keeps data and runs on Drive under
`/content/drive/MyDrive/Thesis/gnn_mech` (`data/` for downloads and caches, `runs/` for run folders).

| Notebook | What it does | Runtime |
|---|---|---|
| `00_quickstart.ipynb` | One short real run (twitch-gamers, `configs/social/quickstart.yaml`, about 15 min), then metrics and training curves | GPU (T4) |
| `01_fetch_datasets.ipynb` | Downloads and parses every dataset into Drive, builds the S2 subgraph caches, shows sizes and S0 structure stats | CPU is fine |
| `02_s2_twitch_gamers.ipynb`, `02_s2_reddit_body.ipynb`, `02_s2_twitch_de.ipynb`, `02_s2_higgs_follow.ipynb` | The S2 α-sweep for one dataset (`configs/sweeps/social_s2_<dataset>.yaml`): dry run, sweep, results table | GPU (T4) |
| `03_s2_results.ipynb` | Collects all finished S2 runs: table per dataset × arch × α, α-curve plots, CSV and PNG in `results/s2/` on Drive | CPU is fine |

## Order

1. Start with `00_quickstart.ipynb` to check that the setup works end to end.
2. `01_fetch_datasets.ipynb` is optional. Each `02_*` notebook downloads its own dataset when it is
   missing. Run `01` first when you want everything prepared in one go, for example on a CPU runtime
   while the GPU sessions are busy. Higgs is large, so it takes a while.
3. Run the four `02_s2_*` notebooks. They can run **in parallel in separate Colab sessions**: each sweep
   writes to its own `exp_name`, so no two sessions share a run folder. Avoid running the same `02`
   notebook in two sessions at once.
4. Run `03_s2_results.ipynb` at any time. It skips unfinished runs.

## Data flow

The first session downloads a dataset into `data/social/<source>/` on Drive. Before training, the
notebook copies that folder to local `/content/data`, copying only missing or newer files, and trains
from there because local disk is much faster than Drive. The first time a subgraph cache
(`processed/<hash>.pt`) is built, it is copied back to Drive, so later sessions skip the partitioning.

## Resume after a disconnect

Run folders and checkpoints live on Drive. Rerun the notebook from the top, then the sweep (or run)
cell: finished runs are skipped and an interrupted run continues from its last `checkpoint.pt`. The
dry-run cell shows which runs are `done`, `partial` or `todo`, and estimates the remaining hours.

## Cells to edit

Only the first code cell (`# EDIT: settings`) needs editing: `BRANCH`, `USE_TOKEN` (set it to `True` and
add a `GITHUB_TOKEN` Colab secret if the repo is private), and the Drive paths. In `01`, the
`DATASETS` cell also lets you choose which datasets to fetch.
