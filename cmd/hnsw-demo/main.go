package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"

	"hnsw-from-scratch/datasets"
	"hnsw-from-scratch/hnsw"
)

type dataset struct {
	vectors [][]float32
	labels  []int
}

type evalResult struct {
	recall float64
	qps    float64
	p50us  int64
	p95us  int64
}

type gloveData struct {
	words   []string
	vectors [][]float32
	dim     int
}

func main() {
	mode := flag.String("mode", "benchmark", "benchmark, two-cluster, glove, sweep, or msweep")
	seed := flag.Int64("seed", 42, "random seed")

	// GloVe/sweep flags
	glovePath := flag.String("glove", "", "path to GloVe file")
	limit := flag.Int("limit", 0, "limit number of vectors (0 = all)")
	queries := flag.Int("queries", 200, "number of query vectors")
	k := flag.Int("k", 10, "top-k")
	m := flag.Int("m", 16, "HNSW M")
	efC := flag.Int("efc", 100, "efConstruction")
	ef := flag.Int("ef", 50, "efSearch")
	metric := flag.String("metric", "l2", "l2 or cosine")
	efList := flag.String("ef-list", "10,25,50,100", "comma-separated efSearch values for sweep mode")
	mList := flag.String("m-list", "8,16,32", "comma-separated M values for msweep mode")
	compare := flag.Bool("compare", false, "compare flat NSW vs HNSW (glove mode)")
	flat := flag.Bool("flat", false, "force single-layer NSW (glove mode)")
	demoWord := flag.String("demo-word", "", "word to query and print nearest neighbors (glove mode)")

	flag.Parse()

	switch *mode {
	case "two-cluster":
		runTwoCluster(*seed)
	case "benchmark":
		runBenchmark(*seed)
	case "glove":
		runGloVe(*seed, *glovePath, *limit, *queries, *k, *m, *efC, *ef, *metric, *compare, *flat, *demoWord)
	case "sweep":
		runSweep(*seed, *glovePath, *limit, *queries, *k, *m, *efC, *efList, *metric)
	case "msweep":
		runMSweep(*seed, *glovePath, *limit, *queries, *k, *mList, *efC, *ef, *metric)
	default:
		fmt.Fprintf(os.Stderr, "unknown mode: %s\n", *mode)
		os.Exit(1)
	}
}

func runTwoCluster(seed int64) {
	nPer := 200
	sep := 8.0
	m := 8

	rng := rand.New(rand.NewSource(seed))
	data := generateTwoClusters(rng, nPer, sep)

	simple := buildNeighborLists(data.vectors, m, false)
	heuristic := buildNeighborLists(data.vectors, m, true)

	sCross, sTotal := crossEdges(simple, data.labels)
	hCross, hTotal := crossEdges(heuristic, data.labels)

	fmt.Printf("Simple selection: %d/%d edges cross clusters (%.3f%%)\n", sCross, sTotal, pct(sCross, sTotal))
	fmt.Printf("Heuristic selection: %d/%d edges cross clusters (%.3f%%)\n", hCross, hTotal, pct(hCross, hTotal))

	fmt.Printf("\nBrute-force top-5 neighbors for query (1,1):\n")
	query := []float32{1.0, 1.0}
	bf := bruteForce(query, data.vectors, 5, hnsw.L2Squared, -1)
	for _, n := range bf {
		fmt.Printf("  idx=%3d label=%d dist2=%.3f\n", n.ID, data.labels[n.ID], n.Dist)
	}
}

func runBenchmark(seed int64) {
	n := 20000
	dim := 64
	queries := 200
	k := 10
	m := 16
	efConstruction := 100
	efSearch := 50

	rng := rand.New(rand.NewSource(seed))
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		vectors[i] = randomVector(rng, dim)
	}

	g, buildTime := buildGraph(vectors, m, efConstruction, hnsw.L2Squared, seed, false)

	queryVecs := make([]indexedQuery, queries)
	for i := 0; i < queries; i++ {
		queryVecs[i] = indexedQuery{vector: randomVector(rng, dim), index: -1}
	}

	ground, bruteRes := bruteForceEval(queryVecs, vectors, k, hnsw.L2Squared)
	res := eval(g, queryVecs, ground, k, efSearch)

	fmt.Printf("Build: n=%d dim=%d M=%d efConstruction=%d time=%s\n", n, dim, m, efConstruction, buildTime)
	fmt.Printf("Brute-force: QPS=%.1f p50=%dus p95=%dus\n", bruteRes.qps, bruteRes.p50us, bruteRes.p95us)
	fmt.Printf("Search: queries=%d k=%d efSearch=%d\n", queries, k, efSearch)
	fmt.Printf("Recall@%d: %.3f\n", k, res.recall)
	fmt.Printf("QPS: %.1f\n", res.qps)
	fmt.Printf("Latency p50=%dus p95=%dus\n", res.p50us, res.p95us)
}

func runGloVe(seed int64, path string, limit, queries, k, m, efC, ef int, metric string, compare, flat bool, demoWord string) {
	if path == "" {
		fmt.Fprintln(os.Stderr, "missing -glove path")
		os.Exit(1)
	}

	data, distance := loadGloVe(path, limit, metric)
	queryRng := rand.New(rand.NewSource(seed + 1))
	sampledQueries := sampleQueriesVectors(data.vectors, queries, queryRng)

	ground, bruteRes := bruteForceEval(sampledQueries, data.vectors, k, distance)

	g, buildTime := buildGraph(data.vectors, m, efC, distance, seed, flat)
	res := eval(g, sampledQueries, ground, k, ef)

	fmt.Printf("Build: n=%d dim=%d time=%s\n", len(data.vectors), data.dim, buildTime)
	fmt.Printf("GloVe: n=%d dim=%d metric=%s M=%d efConstruction=%d flat=%v\n", len(data.vectors), data.dim, metric, m, efC, flat)
	fmt.Printf("Brute-force: QPS=%.1f p50=%dus p95=%dus\n", bruteRes.qps, bruteRes.p50us, bruteRes.p95us)
	fmt.Printf("Search: queries=%d k=%d efSearch=%d\n", queries, k, ef)
	fmt.Printf("Recall@%d: %.3f\n", k, res.recall)
	fmt.Printf("QPS: %.1f\n", res.qps)
	fmt.Printf("Latency p50=%dus p95=%dus\n", res.p50us, res.p95us)

	if compare {
		flatGraph, flatBuild := buildGraph(data.vectors, m, efC, distance, seed+1, true)
		flatRes := eval(flatGraph, sampledQueries, ground, k, ef)
		fmt.Printf("\nCompare (flat vs HNSW)\n")
		fmt.Printf("Flat build time: %s\n", flatBuild)
		fmt.Printf("Flat recall@%d: %.3f QPS: %.1f p50=%dus p95=%dus\n", k, flatRes.recall, flatRes.qps, flatRes.p50us, flatRes.p95us)
		fmt.Printf("HNSW recall@%d: %.3f QPS: %.1f p50=%dus p95=%dus\n", k, res.recall, res.qps, res.p50us, res.p95us)
	}

	if demoWord != "" {
		idx := wordIndex(data.words, demoWord)
		if idx == -1 {
			fmt.Printf("\nDemo word %q not found in GloVe subset\n", demoWord)
			return
		}
		neighbors := g.SearchK(data.vectors[idx], k, ef)
		fmt.Printf("\nNearest neighbors for %q:\n", demoWord)
		for _, n := range neighbors {
			fmt.Printf("  %s (dist=%.4f)\n", data.words[n.ID], n.Dist)
		}
	}
}

func runSweep(seed int64, path string, limit, queries, k, m, efC int, efList, metric string) {
	if path == "" {
		fmt.Fprintln(os.Stderr, "missing -glove path")
		os.Exit(1)
	}

	data, distance := loadGloVe(path, limit, metric)
	queryRng := rand.New(rand.NewSource(seed + 1))
	sampledQueries := sampleQueriesVectors(data.vectors, queries, queryRng)
	ground, _ := bruteForceEval(sampledQueries, data.vectors, k, distance)

	g, buildTime := buildGraph(data.vectors, m, efC, distance, seed, false)

	values := parseIntList(efList)
	if len(values) == 0 {
		fmt.Fprintln(os.Stderr, "-ef-list is empty")
		os.Exit(1)
	}

	fmt.Printf("Build: n=%d dim=%d time=%s\n", len(data.vectors), data.dim, buildTime)
	fmt.Printf("Sweep: n=%d dim=%d metric=%s M=%d efConstruction=%d\n", len(data.vectors), data.dim, metric, m, efC)
	fmt.Printf("efSearch\trecall@%d\tqps\tp50us\tp95us\n", k)
	for _, ef := range values {
		res := eval(g, sampledQueries, ground, k, ef)
		fmt.Printf("%d\t%.3f\t%.1f\t%d\t%d\n", ef, res.recall, res.qps, res.p50us, res.p95us)
	}
}

func runMSweep(seed int64, path string, limit, queries, k int, mList string, efC, ef int, metric string) {
	if path == "" {
		fmt.Fprintln(os.Stderr, "missing -glove path")
		os.Exit(1)
	}

	data, distance := loadGloVe(path, limit, metric)
	queryRng := rand.New(rand.NewSource(seed + 1))
	sampledQueries := sampleQueriesVectors(data.vectors, queries, queryRng)
	ground, _ := bruteForceEval(sampledQueries, data.vectors, k, distance)

	values := parseIntList(mList)
	if len(values) == 0 {
		fmt.Fprintln(os.Stderr, "-m-list is empty")
		os.Exit(1)
	}

	fmt.Printf("MSweep: n=%d dim=%d metric=%s efConstruction=%d efSearch=%d\n", len(data.vectors), data.dim, metric, efC, ef)
	fmt.Printf("M\trecall@%d\tqps\tp50us\tp95us\n", k)
	for i, mv := range values {
		g, buildTime := buildGraph(data.vectors, mv, efC, distance, seed+int64(i), false)
		res := eval(g, sampledQueries, ground, k, ef)
		fmt.Printf("%d\t%.3f\t%.1f\t%d\t%d\t(build %s)\n", mv, res.recall, res.qps, res.p50us, res.p95us, buildTime)
	}
}

func loadGloVe(path string, limit int, metric string) (gloveData, hnsw.DistanceFunc) {
	glo, err := datasets.LoadGloVe(path, limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load glove error: %v\n", err)
		os.Exit(1)
	}

	distance := pickDistance(metric)
	if strings.ToLower(metric) == "cosine" {
		for i := range glo.Vectors {
			hnsw.Normalize(glo.Vectors[i])
		}
	}

	return gloveData{words: glo.Words, vectors: glo.Vectors, dim: glo.Dim}, distance
}

func buildGraph(vectors [][]float32, m, efC int, distance hnsw.DistanceFunc, seed int64, flat bool) (*hnsw.Graph, time.Duration) {
	g := hnsw.NewGraph(m, efC, distance, true, seed)
	g.Flat = flat
	startBuild := time.Now()
	for i := range vectors {
		g.Add(vectors[i])
	}
	return g, time.Since(startBuild)
}

func pickDistance(metric string) hnsw.DistanceFunc {
	switch strings.ToLower(metric) {
	case "cosine":
		return hnsw.CosineDistance
	case "l2":
		return hnsw.L2Squared
	default:
		return hnsw.L2Squared
	}
}

type indexedQuery struct {
	vector []float32
	index  int // position in the original dataset
}

func sampleQueriesVectors(vectors [][]float32, queries int, rng *rand.Rand) []indexedQuery {
	if queries > len(vectors) {
		queries = len(vectors)
	}
	out := make([]indexedQuery, 0, queries)
	perm := rng.Perm(len(vectors))
	for i := 0; i < queries; i++ {
		out = append(out, indexedQuery{vector: vectors[perm[i]], index: perm[i]})
	}
	return out
}

func eval(g *hnsw.Graph, queries []indexedQuery, ground [][]hnsw.Neighbor, k int, efSearch int) evalResult {
	latencies := make([]time.Duration, 0, len(queries))
	matches := 0
	startSearch := time.Now()
	for i, q := range queries {
		start := time.Now()
		// Request k+1 results so we still have k after filtering out the query itself.
		hnswRes := g.SearchK(q.vector, k+1, efSearch)
		latencies = append(latencies, time.Since(start))
		filtered := filterSelf(hnswRes, uint32(q.index), k)
		matches += commonCount(filtered, ground[i])
	}
	searchTime := time.Since(startSearch)

	recall := float64(matches) / float64(len(queries)*k)
	qps := float64(len(queries)) / searchTime.Seconds()

	p50 := percentile(latencies, 50)
	p95 := percentile(latencies, 95)

	return evalResult{recall: recall, qps: qps, p50us: p50.Microseconds(), p95us: p95.Microseconds()}
}

func filterSelf(results []hnsw.Neighbor, selfID uint32, k int) []hnsw.Neighbor {
	out := make([]hnsw.Neighbor, 0, k)
	for _, r := range results {
		if r.ID == selfID {
			continue
		}
		out = append(out, r)
		if len(out) == k {
			break
		}
	}
	return out
}

func bruteForceEval(queries []indexedQuery, vectors [][]float32, k int, distance hnsw.DistanceFunc) ([][]hnsw.Neighbor, evalResult) {
	latencies := make([]time.Duration, 0, len(queries))
	out := make([][]hnsw.Neighbor, len(queries))
	startAll := time.Now()
	for i, q := range queries {
		start := time.Now()
		out[i] = bruteForce(q.vector, vectors, k, distance, q.index)
		latencies = append(latencies, time.Since(start))
	}
	allTime := time.Since(startAll)

	p50 := percentile(latencies, 50)
	p95 := percentile(latencies, 95)

	return out, evalResult{recall: 1.0, qps: float64(len(queries)) / allTime.Seconds(), p50us: p50.Microseconds(), p95us: p95.Microseconds()}
}

func generateTwoClusters(rng *rand.Rand, nPer int, sep float64) dataset {
	vectors := make([][]float32, 0, 2*nPer)
	labels := make([]int, 0, 2*nPer)
	for i := 0; i < nPer; i++ {
		vectors = append(vectors, []float32{float32(rng.NormFloat64()), float32(rng.NormFloat64())})
		labels = append(labels, 0)
	}
	for i := 0; i < nPer; i++ {
		vectors = append(vectors, []float32{float32(rng.NormFloat64() + sep), float32(rng.NormFloat64() + sep)})
		labels = append(labels, 1)
	}
	return dataset{vectors: vectors, labels: labels}
}

func buildNeighborLists(vectors [][]float32, M int, heuristic bool) [][]uint32 {
	N := len(vectors)
	neighbors := make([][]uint32, N)
	for i := 0; i < N; i++ {
		cands := make([]hnsw.Neighbor, 0, N-1)
		for j := 0; j < N; j++ {
			if i == j {
				continue
			}
			d := hnsw.L2Squared(vectors[i], vectors[j])
			cands = append(cands, hnsw.Neighbor{ID: uint32(j), Dist: d})
		}
		if heuristic {
			neighbors[i] = hnsw.SelectNeighborsHeuristic(uint32(i), cands, M, func(a, b uint32) float32 {
				return hnsw.L2Squared(vectors[a], vectors[b])
			})
		} else {
			neighbors[i] = hnsw.SelectNeighborsSimple(uint32(i), cands, M)
		}
	}
	return neighbors
}

func crossEdges(neighbors [][]uint32, labels []int) (int, int) {
	cross := 0
	total := 0
	for i, nbrs := range neighbors {
		for _, j := range nbrs {
			total++
			if labels[i] != labels[j] {
				cross++
			}
		}
	}
	return cross, total
}

func pct(cross, total int) float64 {
	if total == 0 {
		return 0
	}
	return (float64(cross) / float64(total)) * 100
}

func bruteForce(query []float32, vectors [][]float32, k int, distance hnsw.DistanceFunc, excludeIdx int) []hnsw.Neighbor {
	res := make([]hnsw.Neighbor, 0, len(vectors))
	for i, v := range vectors {
		if i == excludeIdx {
			continue
		}
		d := distance(query, v)
		res = append(res, hnsw.Neighbor{ID: uint32(i), Dist: d})
	}
	selectTopK(res, k)
	if len(res) > k {
		res = res[:k]
	}
	return res
}

func selectTopK(res []hnsw.Neighbor, k int) {
	for i := 0; i < k && i < len(res); i++ {
		minIdx := i
		for j := i + 1; j < len(res); j++ {
			if res[j].Dist < res[minIdx].Dist {
				minIdx = j
			}
		}
		res[i], res[minIdx] = res[minIdx], res[i]
	}
}

func commonCount(a, b []hnsw.Neighbor) int {
	set := make(map[uint32]struct{}, len(b))
	for _, n := range b {
		set[n.ID] = struct{}{}
	}
	match := 0
	for _, n := range a {
		if _, ok := set[n.ID]; ok {
			match++
		}
	}
	return match
}

func randomVector(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()
	}
	return v
}

func percentile(latencies []time.Duration, p int) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	idx := int(math.Ceil(float64(p)/100*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func parseIntList(s string) []int {
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var v int
		_, err := fmt.Sscanf(p, "%d", &v)
		if err != nil {
			continue
		}
		out = append(out, v)
	}
	return out
}

func wordIndex(words []string, target string) int {
	for i, w := range words {
		if w == target {
			return i
		}
	}
	return -1
}
