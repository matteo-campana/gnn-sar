# gnn-sar

Spatio-temporal graph neural networks for **sub-annual soil-sealing monitoring**
from Sentinel-1 SAR time series.

The research question comes from the PhD proposal in
[`docs/proposta_dottorato_strutturata_v2.md`](docs/proposta_dottorato_strutturata_v2.md)
(in Italian). Italy's official land-take cartography (ISPRA/SNPA) already uses
Sentinel-1 and Sentinel-2 together; what it does not give is **temporal
granularity** — one map per year — and it costs a great deal of expert
photo-interpretation. This repository is the code side of the attempt to close
that gap: model the territory as a graph of stable land units, let a GNN read
the Sentinel-1 time series over it, and supervise sub-annual predictions with
the annual labels that actually exist.

> **Status: a working research scaffold, not a validated product.**
> Everything here runs end to end and is covered by tests, but only against a
> **synthetic** scenario. There is no Sentinel-1 ingestion and no ISPRA/SNPA
> comparison yet — see [What is not here](#what-is-not-here).

## Install

```bash
poetry install --with dev          # add --extras segmentation for SLIC
```

PyTorch is pinned to `2.6.0`. On a machine without a GPU, install the CPU wheel
first so the CUDA runtime is not pulled in:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

## Quickstart

Train the model and its ablations on a synthetic land-take scenario and print
the comparison:

```bash
gnn-sar demo --models stgnn temporal mlp rf --max-epochs 120
```

Or drive the pieces yourself:

```python
from gnn_sar.data import make_synthetic_scenario
from gnn_sar.models import SpatioTemporalGNN
from gnn_sar.config import ModelConfig
from gnn_sar.training import SpatioTemporalGraphDataModule, SoilSealingModule
import pytorch_lightning as pl

scenario = make_synthetic_scenario(height=80, width=80, n_time=24)
data = SpatioTemporalGraphDataModule(scenario.graph)
data.setup()

module = SoilSealingModule(
    model=SpatioTemporalGNN(ModelConfig(in_channels=data.n_features))
)
pl.Trainer(max_epochs=50).fit(module, datamodule=data)

maps = module.predict_maps(data.batch("test"))
maps["sealed_probability"]        # (T, N) monotone sub-annual sealing map
maps["onset"]                     # (N,)   first composite flagged per node
maps["early_warning"]             # (T, N) "seals within the next h composites"
raster = scenario.geometries.rasterise(maps["sealed_probability"][-1])
```

## The idea in one page

**Nodes are fixed, features move.** Speckle makes a segmentation recomputed at
each acquisition unstable, which would break every temporal edge. So the domain
is segmented **once** — on a multitemporal composite, or from cadastral parcels
— and only the features vary over time. `NodeGeometries` holds that partition.

**Two edge families.** `adjacency` links segments that share a border;
`connectivity` links nearby centroids along an infrastructure corridor. Together
they encode the project's hypothesis: land take propagates along roads and
thickens at the urban fringe, rather than appearing uniformly at random.

**Time is causal.** The spatio-temporal super-graph is block lower-triangular:
`(t-k, i) → (t, i)` and never the reverse. A sub-annual product has to be
producible at time `t` from `t` and its past, or the "detection delay" it
reports is not one the operational chain could ever achieve. The models enforce
this, and a test checks it: perturbing the last composite leaves every earlier
prediction bit-identical.

**Irreversibility is structural, not a penalty.** The detection head does not
predict "is this sealed at `t`". It predicts the per-composite **onset**
probability `q_t`, and the sealed state is the cumulative noisy-OR
`p_t = 1 − Π_{s≤t}(1 − q_s)`, which is non-decreasing for *any* parameters.

**That is also what resolves the annual/sub-annual mismatch.** The probability
that land take happened somewhere inside a year is the same noisy-OR restricted
to that year's composites — so an annual ISPRA/SNPA label supervises a
sub-annual prediction directly, and *where inside the year* the transition sits
is never supervised. A test pins this down: moving a predicted onset between two
composites of the same labelled year does not change the loss at all.

## Layout

| Module | What it does |
| --- | --- |
| `gnn_sar.config` | Dataclass configs, JSON-serialisable, validated at construction |
| `gnn_sar.data.segmentation` | Fixed node geometries: cadastral, regular tiling, or SLIC |
| `gnn_sar.data.features` | Irregular revisit → regular composites; dB; leakage-safe standardisation |
| `gnn_sar.data.graph` | Adjacency and connectivity edges, causal super-graph, geographic splits |
| `gnn_sar.data.targets` | Weak (annual) and strong (dated) supervision targets |
| `gnn_sar.data.synthetic` | A land-take scenario with topological propagation and weak labels |
| `gnn_sar.models.stgnn` | GraphSAGE/GAT spatial stage + causal temporal stage + two heads |
| `gnn_sar.models.baselines` | The same model minus the graph, minus time, and a random forest |
| `gnn_sar.losses.weak` | Multiple-instance annual loss, graph smoothness, single-onset |
| `gnn_sar.metrics` | Detection delay, early-warning lead time, OA/F1/IoU/kappa |
| `gnn_sar.training` | Lightning datamodule, module and the `gnn-sar` CLI |

## What the demo actually reports

One run, seed 0, 400 nodes, 24 composites, `--contrast 0.3` (the change
signature sits close to the speckle), torch 2.6.0:

| model | annual F1 | kappa | detection rate | mean delay | median delay |
| --- | --- | --- | --- | --- | --- |
| `stgnn` — graph + time | 0.80 | 0.76 | 0.86 | +1.9 | +1.0 |
| `temporal` — time only | **0.91** | **0.89** | **0.93** | **+1.4** | **0.0** |
| `mlp` — neither | 0.70 | 0.65 | 0.64 | +3.6 | −1.0 |
| `rf` — random forest, annual only | 0.91 | 0.89 | — | — | — |

Read it honestly: **the graph does not earn its keep here.** The temporal
ablation beats the full model and ties the random forest; what clearly pays is
having *any* temporal context, since the memoryless `mlp` is far behind. The
delays are positive by roughly one composite, and the models do detect some
nodes early — the construction phase is visible before the transition itself.

That is a result about the *simulator*, not about Italy: the generator gives
each node a per-node signature strong enough that its own trajectory nearly
suffices, so spatial message passing mostly blurs it. It is exactly the
comparison the proposal's year-one **feasibility gate** is meant to run, and the
answer it returns on synthetic data is "not yet". Treat the number as a check
that the harness works and reports against itself, not as evidence either way
about the relational hypothesis, which only real Sentinel-1 data can settle.

Reproduce with:

```bash
gnn-sar demo --models stgnn temporal mlp rf --max-epochs 120 \
    --nodes-per-side 20 --n-time 24 --contrast 0.3
```

## What is not here

Named so that nobody mistakes the scaffold for the system:

- **Sentinel-1 ingestion.** No CDSE download, no SNAP graph, no calibration,
  terrain correction, despeckling or InSAR coherence. `make_synthetic_scenario`
  stands in for all of it; the pipeline from `(T, B, H, W)` rasters onward is
  real.
- **ISPRA/SNPA ground truth.** Not redistributable, and not wired in. The
  annual-label contract the code expects is `(Y, N)` with `-1` for unknown.
- **Sentinel-2 fusion.** The feature tensor is source-agnostic, but nothing
  computes optical features or cloud masks.
- **Cadastral geometries.** `NodeGeometries.from_label_image` accepts a
  rasterised parcel layer; rasterising one is left to the user.
- **The siamese U-Net baseline** of the proposal — it works on rasters, not on
  the node graph, and a fake version would be worse than none.
- **EO foundation models** (Prithvi/Presto/Clay) as baselines.

## Tests

```bash
pytest
```

150 tests covering graph-construction invariants, the causality and
monotonicity guarantees, the behaviour of each loss term, the metrics against
scikit-learn, and an end-to-end training cycle. One is skipped unless the
optional `segmentation` extra is installed.

## Licence

MIT, for the code only — see [LICENSE.md](LICENSE.md), which also lists the
separate terms attached to the Copernicus and ISPRA/SNPA data this project is
meant to consume.
