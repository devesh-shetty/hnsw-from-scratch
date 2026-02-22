package hnsw

import "testing"

func TestSelectNeighborsSimple(t *testing.T) {
	cands := []Neighbor{{ID: 1, Dist: 3}, {ID: 2, Dist: 1}, {ID: 3, Dist: 2}}
	got := SelectNeighborsSimple(0, cands, 2)
	if len(got) != 2 {
		t.Fatalf("expected 2 neighbors, got %d", len(got))
	}
	if got[0] != 2 || got[1] != 3 {
		t.Fatalf("expected [2 3], got %v", got)
	}
}

func TestSelectNeighborsHeuristic(t *testing.T) {
	dist := map[[2]uint32]float32{
		{1, 0}: 1,
		{2, 0}: 2,
		{3, 0}: 3,
		{2, 1}: 0.5,
		{3, 1}: 10,
		{3, 2}: 10,
	}
	distance := func(a, b uint32) float32 {
		if v, ok := dist[[2]uint32{a, b}]; ok {
			return v
		}
		if v, ok := dist[[2]uint32{b, a}]; ok {
			return v
		}
		return 0
	}

	cands := []Neighbor{{ID: 1, Dist: 1}, {ID: 2, Dist: 2}, {ID: 3, Dist: 3}}
	got := SelectNeighborsHeuristic(0, cands, 2, distance)
	if len(got) != 2 {
		t.Fatalf("expected 2 neighbors, got %d", len(got))
	}
	if got[0] != 1 || got[1] != 3 {
		t.Fatalf("expected [1 3], got %v", got)
	}
}

func TestFlatRandLevel(t *testing.T) {
	g := NewGraph(16, 100, L2Squared, true, 42)
	g.Flat = true
	for i := 0; i < 10; i++ {
		if lvl := g.randLevel(); lvl != 0 {
			t.Fatalf("expected level 0 for flat graph, got %d", lvl)
		}
	}
}
