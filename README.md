# hnsw-from-scratch

A minimal [HNSW](https://arxiv.org/abs/1603.09320) implementation in Go, built from first principles. The companion blog post is [Building HNSW From First Principles](https://deveshshetty.com/blog/hnsw-from-scratch/).

The core library is ~430 lines of Go (excluding tests). No dependencies beyond the standard library.

## What is in here

```text
hnsw/
  types.go      -- Node, Graph, Neighbor structs
  graph.go      -- Insert, SearchK, searchLayer, pruning
  select.go     -- Simple and heuristic neighbor selection
  distance.go   -- L2Squared, CosineDistance, Normalize
  heap.go       -- Min/max heaps for beam search
  hnsw_test.go  -- Recall test on small random data
  select_test.go -- Heuristic selection unit test

datasets/
  glove.go      -- GloVe text file loader
  glove_test.go -- Loader test with tiny fixture

cmd/hnsw-demo/
  main.go       -- CLI for benchmarks, sweeps, two-cluster demo, GloVe runs
```

## Design decisions

- **No generics.** Vectors are `[]float32`, IDs are `uint32`. Simpler to read and matches what production systems do.
- **No concurrency.** Single-threaded insert and search. The blog focuses on the algorithm, not the locking.
- **No persistence.** The graph lives in memory. No serialization, no WAL.
- **Heuristic neighbor selection** (Algorithm 4 from the paper) is on by default. It prevents cluster disconnection.
- **Layer 0 uses `2M` connections**, upper layers use `M`. This matches the paper and improves recall at the densest layer.
- **Epoch-based visited tracking.** Instead of allocating a `[]bool` per search call, the graph maintains a `[]uint64` with a monotonic epoch counter. Reset is O(1).
- **Back-edge removal on prune.** When a node's connection list is pruned, back-edges from removed neighbors are also cleaned up. This deviates from the paper but keeps adjacency lists symmetric.
- **Flat NSW mode.** Set `g.Flat = true` to force all nodes onto layer 0 (no hierarchy). This is how the FlatNav comparison works.

## Quick start

Run the tests:

```bash
go test ./...
```

Two-cluster heuristic demo:

```bash
go run ./cmd/hnsw-demo -mode=two-cluster
```

```text
Simple selection: 0/3200 edges cross clusters (0.000%)
Heuristic selection: 10/3200 edges cross clusters (0.312%)

Brute-force top-5 neighbors for query (1,1):
  idx= 28 label=0 dist2=0.003
  idx=162 label=0 dist2=0.020
  idx=107 label=0 dist2=0.042
  idx= 97 label=0 dist2=0.059
  idx= 96 label=0 dist2=0.117
```

Random-vector benchmark (no external data needed):

```bash
go run ./cmd/hnsw-demo -mode=benchmark
```

## GloVe runs

Download [GloVe](https://nlp.stanford.edu/projects/glove/) (the 6B set) and unzip. Then:

**Baseline with brute-force comparison and nearest-neighbor demo:**

```bash
go run ./cmd/hnsw-demo \
  -mode=glove \
  -glove ~/Datasets/glove.6B.100d.txt \
  -limit 20000 \
  -queries 200 \
  -k 10 \
  -m 16 \
  -efc 100 \
  -ef 50 \
  -metric l2 \
  -compare \
  -demo-word king
```

This prints brute-force vs HNSW recall/QPS, a flat NSW vs HNSW comparison (when `-compare` is set), and the nearest neighbors for "king".

**efSearch sweep:**

```bash
go run ./cmd/hnsw-demo \
  -mode=sweep \
  -glove ~/Datasets/glove.6B.100d.txt \
  -limit 20000 \
  -queries 200 \
  -k 10 \
  -m 16 \
  -efc 100 \
  -metric l2 \
  -ef-list 10,25,50,100,200
```

**M sweep:**

```bash
go run ./cmd/hnsw-demo \
  -mode=msweep \
  -glove ~/Datasets/glove.6B.100d.txt \
  -limit 20000 \
  -queries 200 \
  -k 10 \
  -efc 100 \
  -ef 50 \
  -metric l2 \
  -m-list 8,16,32
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `-mode` | `benchmark` | `benchmark`, `two-cluster`, `glove`, `sweep`, or `msweep` |
| `-seed` | `42` | Random seed for reproducibility |
| `-glove` | | Path to a GloVe text file |
| `-limit` | `0` (all) | Max vectors to load |
| `-queries` | `200` | Number of query vectors |
| `-holdout` | `0` | Hold out N vectors for the query set (glove/sweep/msweep) |
| `-k` | `10` | Top-k neighbors |
| `-m` | `16` | HNSW M parameter (max connections per layer) |
| `-efc` | `100` | efConstruction (build-time search depth) |
| `-ef` | `50` | efSearch (query-time search depth) |
| `-metric` | `l2` | `l2` or `cosine` |
| `-ef-list` | `10,25,50,100` | Comma-separated efSearch values for sweep mode |
| `-m-list` | `8,16,32` | Comma-separated M values for msweep mode |
| `-compare` | `false` | Also build a flat NSW graph and compare |
| `-flat` | `false` | Force single-layer NSW |
| `-demo-word` | | Word to query and print nearest neighbors |

## Notes

- Queries are sampled from the indexed set. Self-matches are excluded from recall scoring.
- Use `-holdout` to split the dataset into index and query sets. Queries are drawn from the held-out set.
- `-metric cosine` normalizes all vectors before indexing.
- Brute-force ground truth is computed over the loaded subset. Large `-limit` values make this slow.
- GloVe runs print a post-build memory snapshot (heapAlloc, heapSys, sys, and RSS when available).
- Defaults are intentionally small to keep runs fast. Use `-limit 20000` for quick iteration, the full 1.2M GloVe set for publication-quality numbers.

## References

- Malkov & Yashunin, [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) (IEEE TPAMI 2020)
- Munyampirwa, Lakshman, Coleman, [Down with the Hierarchy](https://arxiv.org/abs/2412.01940) (VecDB@ICML 2025)
- [hnswlib](https://github.com/nmslib/hnswlib) (reference C++ implementation by Malkov)
