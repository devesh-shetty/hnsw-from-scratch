# hnsw-from-scratch

Minimal HNSW implementation in Go with a small demo harness and a GloVe loader.

## Quick start

```bash
go test ./...
```

Two-cluster heuristic demo (matches the blog output):

```bash
go run ./cmd/hnsw-demo -mode=two-cluster
```

Sample output:

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

Quick benchmark (random vectors, small default sizes):

```bash
go run ./cmd/hnsw-demo -mode=benchmark
```

Sample output:

```text
Build: n=20000 dim=64 M=16 efConstruction=100 time=7.802839375s
Search: queries=200 k=10 efSearch=50
Recall@10: 0.682
QPS: 9216.1
Latency p50=105us p95=128us
```

## GloVe runs

Evaluate on a local GloVe file (use a subset for iteration):

```bash
go run ./cmd/hnsw-demo -mode=glove -glove ~/Datasets/glove.6B.100d.txt -limit 50000 -queries 200 -k 10 -m 16 -efc 100 -ef 50 -metric l2
```

Sweep `efSearch` values on the same index:

```bash
go run ./cmd/hnsw-demo -mode=sweep -glove ~/Datasets/glove.6B.100d.txt -limit 50000 -queries 200 -k 10 -m 16 -efc 100 -metric l2 -ef-list 10,25,50,100,200
```

Notes:

- `metric=cosine` normalizes vectors before indexing.
- Brute-force ground truth is computed over the loaded subset, so large `-limit` values will be slow.
- Defaults are intentionally small to keep runs fast.
