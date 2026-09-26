---
license: cc-by-4.0
language:
  - en
tags:
  - connectomics
  - neuroscience
  - drosophila
  - connectome
  - graph
pretty_name: MaleCNS 166k — the complete fruit fly connectome as typed arrays
size_categories:
  - 10M<n<100M
---

# Fly Connectome — MaleCNS 166k

**The whole fruit fly central nervous system, as plain typed arrays.**
166,700 neurons, 25,582,938 directed connections, 124,177,617 synaptic contacts —
raw signed weights, nothing normalized, in a destination-major CSR you can `np.fromfile`
and use in about four lines.

This is the **complete** graph. If you want only the 49,393-neuron central-brain subset
that the Fly LLM language models run on, with measured soma positions, that is
[fly-connectome-49k](https://huggingface.co/datasets/fernandofernandes/fly-connectome-49k).
The two use the same conventions and compose.

> Measured biological connectivity. Nothing here was learned or fitted.

## Contents — 209.5 MB

| File | Shape | dtype | What it is |
|---|---|---|---|
| `graph/edges_offsets.i32` | 166,701 | int32 | CSR row offsets; **row = postsynaptic neuron** |
| `graph/edges_source.i32` | 25,582,938 | int32 | presynaptic neuron index |
| `graph/edges_weight.f32` | 25,582,938 | float32 | **signed** synaptic weight |
| `neurons/body_id.i64` | 166,700 | int64 | MaleCNS body IDs |
| `neurons/superclass_index.i32` + `superclass_labels.json` | 166,700 / 27 | int32 | superclass |
| `neurons/cell_type_index.i32` + `cell_type_labels.json` | 166,700 / 11,752 | int32 | cell type |
| `populations/{retina,lamina,descending,sugar}.i32` | 3,335 / 7,114 / 1,314 / 23 | int32 | named input and output sets |
| `derived/hops_from_retina.i32`, `hops_to_descending.i32` | 166,700 | int32 | hop counts (**computed here, not upstream**) |
| `manifest.json` | — | — | shapes, dtypes, SHA-256 of every file, provenance |

**15,768,940 excitatory and 9,813,998 inhibitory** edges, none zero. Mean in-degree 153.5,
median 98, maximum 11,526. Weight magnitudes run from 0 to 712.5 with a median of 0.55.

The largest superclasses: `ol_intrinsic` 89,403 · `cb_intrinsic` 32,164 ·
`vnc_intrinsic` 13,161 · `visual_projection` 9,201 · `vnc_sensory` 6,370 ·
`ol_sensory` 6,098 · `cb_sensory` 4,868 · `ascending_neuron` 1,846 ·
`descending_neuron` 1,314 · `vnc_motor` 708.

## Load it

```python
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(snapshot_download('fernandofernandes/fly-connectome-malecns-166k', repo_type='dataset'))
offsets = np.fromfile(root / 'graph/edges_offsets.i32', dtype=np.int32)   # 166,701
source  = np.fromfile(root / 'graph/edges_source.i32',  dtype=np.int32)   # 25,582,938
weight  = np.fromfile(root / 'graph/edges_weight.f32',  dtype=np.float32)

# Everything presynaptic to neuron 12345, as (index, signed weight) pairs:
lo, hi = offsets[12345], offsets[12346]
incoming = list(zip(source[lo:hi], weight[lo:hi]))
```

As a sparse matrix, `W[postsynaptic, presynaptic]`:

```python
from scipy.sparse import csr_matrix
W = csr_matrix((weight, source, offsets), shape=(166700, 166700))
```

Drive it like a rate network — visual input into `populations/retina.i32`, read motor
intent out of `populations/descending.i32`:

```python
retina = np.fromfile(root / 'populations/retina.i32', dtype=np.int32)
descending = np.fromfile(root / 'populations/descending.i32', dtype=np.int32)

x = np.zeros(166700, dtype=np.float32)
for _ in range(steps):
    drive = np.zeros_like(x)
    drive[retina] = stimulus                      # your input, however you encode it
    x = 0.9 * np.tanh(W @ x + drive) + 0.1 * x    # your dynamics, this is only an example
motor = x[descending]
```

The recurrence above is **an example, not part of the dataset**. The connectome fixes the
wiring; every dynamical choice — gains, leak, nonlinearity, timestep — is yours.

## Verify it

Every file's SHA-256 is in `manifest.json`.
[`verify_connectome_package.py`](https://github.com/fernando-neto-ai/fly-wordbrain/blob/main/scripts/verify_connectome_package.py)
runs 17 structural checks with nothing but NumPy, and works on both this dataset and the
49k subset:

```
All 17 checks passed: 166,700 neurons, 25,582,938 edges, 11,752 cell types, 4 named populations.
```

## What the weights mean, precisely

A weight is a **signed synaptic count**: magnitude from the flat connectome at minimum
confidence **0.5**, sign from the neurotransmitter prediction. It is **not** a conductance,
not a measured physiological strength, and not calibrated across cell types. No
normalization of any kind has been applied — `normalization_applied` is `false` in the
manifest, and any scaling you need is yours to choose and to record.

Edges are **not coalesced**. Cell type is an empty label where the source carries no
official type. The two `derived/` hop arrays were computed by this project, not by the
upstream release, and are `-1` where a neuron is unreachable.

**Soma coordinates are not included here.** The
[49k subset](https://huggingface.co/datasets/fernandofernandes/fly-connectome-49k)
carries measured soma positions for its neurons.

## Provenance

Assembled from the three publicly hosted MaleCNS v1.0 flat-connectome files, via
[Doomfly](https://github.com/nftechie/doomfly) at commit `71ecf53d`:

| Upstream file | Bytes | SHA-256 |
|---|---:|---|
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | `e35da783…` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | `95c92892…` |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | `2177e246…` |

Full URLs and digests are in `manifest.json` under `provenance.upstream_files`, so you can
rebuild this package from the original release and check it against ours.

## Licence and attribution

**CC BY 4.0.** Connectome data: FlyEM / HHMI Janelia Research Campus, University of
Cambridge, MRC Laboratory of Molecular Biology, and Google Research. Packaging code MIT.

Used by [fly-wordbrain](https://github.com/fernando-neto-ai/fly-wordbrain); hear the
49k subset write and read a story aloud at
[Fly Recital](https://huggingface.co/spaces/fernandofernandes/fly-recital).
