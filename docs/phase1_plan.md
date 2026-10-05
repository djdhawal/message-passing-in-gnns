# Phase 1: shared training and measurement setup

Goal: one command trains any point on the attention/MPNN spectrum the right way,
measures it with every diagnostic, and writes results to disk. Later phases are
then config files plus Colab compute.

## Bugs in the pilot notebooks this phase fixes

| # | Problem | Fix |
|---|---|---|
| 1 | Categorical atom/bond codes fed through `nn.Linear(x.float())` | One embedding table per categorical column (`encoders.py`) |
| 2 | Jacobian taken w.r.t. raw integer features | Jacobian w.r.t. embedded input `h0` (`measure/jacobian.py`) |
| 3 | Mechanistic GPS reuses one LayerNorm twice per layer | Separate norms per block (`layers.py`) |
| 4 | No positional encoding in Experiment 3 models | RWSE for every model (`data.py`, `encoders.py`) |
| 5 | Entropy computed on head-averaged attention | Per-head entropy (`measure/entropy.py`) |
| 6 | Sum pooling (output scale tied to graph size) | Mean pooling everywhere |
| 7 | Jacobian needs one backward pass per hidden dim | Vectorized (`measure/jacobian.py`) |
| 8 | Jacobian pairs store only R and J | Also store hop distance, graph size, graph id |
| 9 | GINE without residual connections | Residuals in every layer |

## Layout

```
gnn_mech/
  config.py        Config dataclasses, YAML loading, dotted overrides        [done]
  results.py       RunDir: run folder, JSON, checkpoints                     [done]
  data.py          dataset loading + RWSE + cached per-graph structure stats
  encoders.py      categorical atom/bond embeddings + PE projection
  layers.py        GlobalAttn (per-head weight cache), HybridLayer (alpha blend)
  models.py        HybridGNN, GCNBaseline, build_model
  train.py         training loop, evaluation, resume
  run.py           CLI entry point
  measure/
    __init__.py    run_all()
    resistance.py  effective resistance, LCC, hop distances
    jacobian.py    vectorized sensitivity
    entropy.py     per-head normalized attention entropy
configs/           one YAML per experiment
tests/             CPU-only pytest
notebooks/colab_driver.ipynb
notebooks/pilot/   original notebooks, unchanged
```

## Interface contract

Every module codes against these signatures. Do not change them without
updating this file.

### Batches (`data.py`)

Each graph is a PyG `Data` with:

- `x`: `LongTensor [N, F_node]`: categorical node codes (Peptides: 9 OGB atom columns)
- `edge_index`: `LongTensor [2, E]`
- `edge_attr`: `LongTensor [E, F_edge]`: categorical edge codes (Peptides: 3 OGB bond columns)
- `pe`: `FloatTensor [N, pe_dim]`: RWSE (absent when `pe_dim == 0`)
- `y`: `[1, out_dim]` float targets (multi-label for Peptides-func)
- `graph_id`: `LongTensor [1]`: index of the graph within its split

```python
@dataclass
class DataInfo:
    node_vocab: list[int]     # vocabulary size of each categorical node column
    edge_vocab: list[int]     # vocabulary size of each categorical edge column
    out_dim: int
    pe_dim: int
    task: str                 # "multilabel" (AP metric) or "regression" (MAE metric)

def load_dataset(cfg: DataConfig) -> tuple[Dataset, Dataset, Dataset, DataInfo]
    # train, val, test. "synthetic-tiny" builds small random molecule-like graphs
    # with the same schema, for tests and CPU smoke runs (no download).

def structure_stats(dataset, split: str, cfg: DataConfig) -> dict[str, np.ndarray]
    # per-graph arrays indexed by graph_id, cached to disk as .npz:
    # "n_nodes", "n_edges", "avg_resistance", "diameter" (LCC), "lcc_size"
```

### Encoders (`encoders.py`)

```python
class CategoricalEncoder(nn.Module):   # sum of one nn.Embedding per column
    def __init__(self, vocab: list[int], dim: int)
    def forward(self, codes: LongTensor) -> FloatTensor   # [*, dim]

class InputEncoder(nn.Module):         # node codes + RWSE -> h0 [N, hidden]
    def __init__(self, info: DataInfo, hidden: int)
    def forward(self, batch) -> FloatTensor
```

### Layers and models (`layers.py`, `models.py`)

```python
class GlobalAttn(nn.Module):
    # dense multi-head self-attention within each graph (to_dense_batch + key padding mask)
    cache: bool                     # when True, forward stores the next two attributes
    last_weights: Tensor | None     # [B, heads, Nmax, Nmax], per head (NOT head-averaged)
    last_mask: Tensor | None        # [B, Nmax] bool, True for real nodes
    def forward(self, x, batch) -> Tensor   # [N, hidden], attention output only (no residual)

class HybridLayer(nn.Module):
    # h_loc = Norm_loc(GINE(x, e)); h_glb = Norm_glb(Attn(x))
    # x = x + (1-alpha)*h_loc + alpha*h_glb
    # x = Norm_ffn(x + FFN(x))
    # alpha == 0: no attention module is built. alpha == 1: no GINE module is built.

class HybridGNN(nn.Module):
    def __init__(self, cfg: ModelConfig, info: DataInfo)
    def embed_inputs(self, batch) -> Tensor                      # h0 [N, hidden]
    def node_embeddings_from_h0(self, h0, batch) -> Tensor       # [N, hidden], final node reps
    def node_embeddings(self, batch) -> Tensor
    def forward(self, batch) -> Tensor                           # logits [B, out_dim]
    def attention_layers(self) -> list[GlobalAttn]               # [] when alpha == 0
    def set_attention_cache(self, on: bool) -> None

class GCNBaseline(nn.Module):        # Tönshoff-style GCN: residual, BN, mean pool, MLP head
    # same embed_inputs / node_embeddings_from_h0 / node_embeddings / forward / attention_layers API

def build_model(cfg: ModelConfig, info: DataInfo) -> nn.Module
def count_parameters(model) -> int
```

`node_embeddings_from_h0` must be a pure function of `h0` (plus fixed graph
structure in `batch`), so Jacobians w.r.t. `h0` are meaningful. Models must be
deterministic in `eval()` mode.

### Training (`train.py`)

```python
def resolve_device(name: str) -> torch.device
def set_seed(seed: int) -> None
def evaluate(model, loader, info: DataInfo, device) -> dict   # {"ap": ...} or {"mae": ...}, plus "loss"
def train(cfg: Config, model, train_ds, val_ds, test_ds, info: DataInfo, run: RunDir) -> dict
    # AdamW, linear warmup then cosine decay, grad clipping, best-val checkpoint (best.pt),
    # resumable checkpoint.pt every cfg.train.checkpoint_every epochs, history.json each epoch.
    # Returns {"best_epoch", "val", "test", "n_params", "epochs_run", "seconds"} and writes metrics.json.
```

### Measurements (`measure/`)

```python
# resistance.py
def lcc_nodes(edge_index, num_nodes) -> np.ndarray
def effective_resistance_matrix(edge_index, num_nodes, nodes=None) -> np.ndarray   # pinv(L) on given nodes
def hop_distance_matrix(edge_index, num_nodes, nodes=None) -> np.ndarray
def avg_effective_resistance(edge_index, num_nodes) -> float

# jacobian.py
def jacobian_norms(model, data, pairs: list[tuple[int, int]], device) -> np.ndarray
    # ||d h_v / d h0_u||_F for each (u, v), vectorized over the hidden dim
def measure_jacobians(model, dataset, cfg: MeasureConfig, device) -> list[dict]
    # rows: {"graph_id", "u", "v", "resistance", "hops", "n_nodes", "jacobian"}
    # pairs stratified across each graph's R quantiles, LCC only

# entropy.py
def attention_entropy(weights, mask) -> np.ndarray     # per graph, layer, head
def measure_entropy(model, dataset, cfg: MeasureConfig, device) -> list[dict]
    # rows: {"graph_id", "n_nodes", "layer", "head", "entropy", "entropy_norm"}
    # [] for models without attention

# __init__.py
def run_all(model, dataset, cfg: MeasureConfig, device) -> dict
    # {"jacobian": [...], "entropy": [...], "summary": {...}}; summary holds
    # log-log slope of jacobian vs resistance and size-entropy correlations
```

### CLI (`run.py`)

```
python -m gnn_mech.run --config configs/peptides_func_gps.yaml --seed 0 [key=value ...]
```

Loads config, seeds, loads data, builds model, trains (resuming if a checkpoint
exists, skipping if `metrics.json` exists unless `--force`), reloads `best.pt`,
runs measurements, writes `measurements.json`.

## Build order and exit criteria

1. Skeleton, config, results folders.
2. `data.py`, `encoders.py`, structure cache.
3. `layers.py`, `models.py`.
4. `train.py`, resume.
5. `measure/*` with analytic tests (path graph R = hop distance, complete graph R = 2/n,
   GINE Jacobian exactly zero beyond L hops, uniform attention entropy = log N).
6. Colab driver notebook.
7. GPU equivalence check: alpha = 0.5 matches or beats the pilot LRGB notebook's
   0.652 test AP on one seed; alpha = 0 and alpha = 1 train without problems.

Phase 1 is done when all tests pass, step 7 holds, and a measurement run on a
trained model writes the full Jacobian and entropy tables.
