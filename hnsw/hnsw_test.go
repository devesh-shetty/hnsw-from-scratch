package hnsw

import (
	"math/rand"
	"testing"
)

func TestSmallRecall(t *testing.T) {
	g := NewGraph(8, 64, L2Squared, true, 42)
	vectors := make([][]float32, 0, 200)
	for i := 0; i < 200; i++ {
		v := randomVector(8)
		vectors = append(vectors, v)
		g.Add(v)
	}

	q := vectors[0]
	k := 5
	hnswRes := g.SearchK(q, k, 32)
	bfRes := bruteForce(q, vectors, k)
	if recallAtK(hnswRes, bfRes, k) < 1.0 {
		t.Fatalf("expected recall@%d = 1.0 on tiny dataset", k)
	}
}

func randomVector(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}

func bruteForce(query []float32, vectors [][]float32, k int) []Neighbor {
	res := make([]Neighbor, 0, len(vectors))
	for i, v := range vectors {
		d := L2Squared(query, v)
		res = append(res, Neighbor{ID: uint32(i), Dist: d})
	}
	// simple partial sort
	for i := 0; i < k && i < len(res); i++ {
		minIdx := i
		for j := i + 1; j < len(res); j++ {
			if res[j].Dist < res[minIdx].Dist {
				minIdx = j
			}
		}
		res[i], res[minIdx] = res[minIdx], res[i]
	}
	if len(res) > k {
		res = res[:k]
	}
	return res
}

func recallAtK(hnswRes, bfRes []Neighbor, k int) float64 {
	set := make(map[uint32]struct{}, len(bfRes))
	for _, n := range bfRes {
		set[n.ID] = struct{}{}
	}
	match := 0
	for _, n := range hnswRes {
		if _, ok := set[n.ID]; ok {
			match++
		}
	}
	return float64(match) / float64(k)
}
