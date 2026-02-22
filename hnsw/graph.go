package hnsw

import (
	"math"
	"math/rand"
	"sort"
)

func NewGraph(m int, efConstruction int, distance DistanceFunc, useHeuristic bool, seed int64) *Graph {
	if m < 2 {
		m = 2
	}
	if efConstruction <= 0 {
		efConstruction = 100
	}
	if distance == nil {
		distance = L2Squared
	}
	ml := 1.0 / math.Log(float64(m))
	if seed == 0 {
		seed = 42
	}
	rng := rand.New(rand.NewSource(seed))
	g := &Graph{
		M:              m,
		M0:             2 * m,
		EfConstruction: efConstruction,
		ML:             ml,
		Distance:       distance,
		UseHeuristic:   useHeuristic,
		Seed:           seed,
		rng:            rng,
	}
	return g
}

func (g *Graph) randLevel() int {
	if g.Flat {
		return 0
	}
	for {
		u := g.rng.Float64()
		if u > 0 {
			r := -math.Log(u) * g.ML
			if r < 0 {
				return 0
			}
			return int(r)
		}
	}
}

func (g *Graph) Add(vector []float32) uint32 {
	id := uint32(len(g.Nodes))
	level := g.randLevel()

	n := Node{
		ID:          id,
		Vector:      vector,
		Level:       level,
		Connections: make([][]uint32, level+1),
	}
	g.Nodes = append(g.Nodes, n)

	if len(g.Nodes) == 1 {
		g.EntryPoint = id
		g.MaxLevel = level
		return id
	}

	ep := g.EntryPoint
	maxLevel := g.MaxLevel

	for l := maxLevel; l > level; l-- {
		best := g.searchLayer(vector, ep, 1, l)
		if len(best) > 0 {
			ep = best[0].ID
		}
	}

	for l := min(maxLevel, level); l >= 0; l-- {
		cands := g.searchLayer(vector, ep, g.EfConstruction, l)
		targetM := g.M
		if l == 0 {
			targetM = g.M0
		}
		neighbors := g.selectNeighbors(id, cands, targetM, l)

		g.Nodes[id].Connections[l] = append(g.Nodes[id].Connections[l], neighbors...)
		for _, nb := range neighbors {
			g.addConnection(nb, id, l)
			g.pruneConnections(nb, l)
		}

		if len(neighbors) > 0 {
			ep = neighbors[0]
		}
	}

	if level > g.MaxLevel {
		g.EntryPoint = id
		g.MaxLevel = level
	}

	return id
}

func (g *Graph) SearchK(query []float32, k int, efSearch int) []Neighbor {
	if len(g.Nodes) == 0 {
		return nil
	}
	if k <= 0 {
		return nil
	}
	if efSearch < k {
		efSearch = k
	}
	ep := g.EntryPoint
	for l := g.MaxLevel; l > 0; l-- {
		best := g.searchLayer(query, ep, 1, l)
		if len(best) > 0 {
			ep = best[0].ID
		}
	}
	res := g.searchLayer(query, ep, efSearch, 0)
	sort.Slice(res, func(i, j int) bool { return res[i].Dist < res[j].Dist })
	if len(res) > k {
		res = res[:k]
	}
	return res
}

func (g *Graph) searchLayer(query []float32, entry uint32, ef int, level int) []Neighbor {
	n := len(g.Nodes)
	g.visitEpoch++
	if len(g.visited) < n {
		g.visited = make([]uint64, n)
	}
	epoch := g.visitEpoch

	candidates := make(minHeap, 0, ef)
	results := make(maxHeap, 0, ef)

	entryDist := g.Distance(query, g.Nodes[entry].Vector)
	pushMin(&candidates, Neighbor{ID: entry, Dist: entryDist})
	pushMax(&results, Neighbor{ID: entry, Dist: entryDist})
	g.visited[entry] = epoch

	for candidates.Len() > 0 {
		cand := popMin(&candidates)
		worst := peekMax(&results)
		if cand.Dist > worst.Dist {
			break
		}
		for _, nb := range g.Nodes[cand.ID].Connections[level] {
			if g.visited[nb] == epoch {
				continue
			}
			g.visited[nb] = epoch
			d := g.Distance(query, g.Nodes[nb].Vector)
			if results.Len() < ef || d < worst.Dist {
				pushMin(&candidates, Neighbor{ID: nb, Dist: d})
				pushMax(&results, Neighbor{ID: nb, Dist: d})
				if results.Len() > ef {
					popMax(&results)
				}
				worst = peekMax(&results)
			}
		}
	}

	out := make([]Neighbor, results.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = popMax(&results)
	}
	return out
}

func (g *Graph) selectNeighbors(queryID uint32, candidates []Neighbor, M int, level int) []uint32 {
	if g.UseHeuristic {
		return SelectNeighborsHeuristic(queryID, candidates, M, func(a, b uint32) float32 {
			return g.Distance(g.Nodes[a].Vector, g.Nodes[b].Vector)
		})
	}
	return SelectNeighborsSimple(queryID, candidates, M)
}

func (g *Graph) addConnection(from, to uint32, level int) {
	list := g.Nodes[from].Connections[level]
	for _, id := range list {
		if id == to {
			return
		}
	}
	g.Nodes[from].Connections[level] = append(g.Nodes[from].Connections[level], to)
}

func (g *Graph) pruneConnections(nodeID uint32, level int) {
	max := g.M
	if level == 0 {
		max = g.M0
	}
	list := g.Nodes[nodeID].Connections[level]
	if len(list) <= max {
		return
	}

	cands := make([]Neighbor, 0, len(list))
	for _, id := range list {
		d := g.Distance(g.Nodes[nodeID].Vector, g.Nodes[id].Vector)
		cands = append(cands, Neighbor{ID: id, Dist: d})
	}
	selected := g.selectNeighbors(nodeID, cands, max, level)

	selectedSet := make(map[uint32]struct{}, len(selected))
	for _, id := range selected {
		selectedSet[id] = struct{}{}
	}

	// Remove back edges for pruned neighbors.
	for _, id := range list {
		if _, ok := selectedSet[id]; ok {
			continue
		}
		g.removeConnection(id, nodeID, level)
	}

	g.Nodes[nodeID].Connections[level] = selected
}

func (g *Graph) removeConnection(from, to uint32, level int) {
	list := g.Nodes[from].Connections[level]
	for i, id := range list {
		if id == to {
			list[i] = list[len(list)-1]
			g.Nodes[from].Connections[level] = list[:len(list)-1]
			return
		}
	}
}
