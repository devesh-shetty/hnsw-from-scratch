package hnsw

import "sort"

// SelectNeighborsSimple returns the M closest candidates to the query.
func SelectNeighborsSimple(queryID uint32, candidates []Neighbor, M int) []uint32 {
	if len(candidates) == 0 || M <= 0 {
		return nil
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Dist < candidates[j].Dist })
	if len(candidates) > M {
		candidates = candidates[:M]
	}
	out := make([]uint32, len(candidates))
	for i, c := range candidates {
		out[i] = c.ID
	}
	return out
}

// SelectNeighborsHeuristic implements the diversity heuristic from the HNSW paper.
// A candidate is kept only if it is closer to the query than to any already selected neighbor.
func SelectNeighborsHeuristic(queryID uint32, candidates []Neighbor, M int, distance func(a, b uint32) float32) []uint32 {
	if len(candidates) == 0 || M <= 0 {
		return nil
	}
	_ = queryID
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Dist < candidates[j].Dist })
	selected := make([]uint32, 0, M)
	rejected := make([]Neighbor, 0, len(candidates))
	for _, c := range candidates {
		ok := true
		for _, s := range selected {
			if distance(c.ID, s) < c.Dist {
				ok = false
				break
			}
		}
		if ok {
			selected = append(selected, c.ID)
			if len(selected) >= M {
				return selected
			}
		} else {
			rejected = append(rejected, c)
		}
	}

	// If we don't have enough neighbors, fill with closest rejected candidates.
	if len(selected) < M && len(rejected) > 0 {
		sort.Slice(rejected, func(i, j int) bool { return rejected[i].Dist < rejected[j].Dist })
		for _, c := range rejected {
			selected = append(selected, c.ID)
			if len(selected) >= M {
				break
			}
		}
	}
	return selected
}
