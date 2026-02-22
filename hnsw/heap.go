package hnsw

import "container/heap"

type minHeap []Neighbor

type maxHeap []Neighbor

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].Dist < h[j].Dist }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) {
	*h = append(*h, x.(Neighbor))
}
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].Dist > h[j].Dist }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x interface{}) {
	*h = append(*h, x.(Neighbor))
}
func (h *maxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func pushMin(h *minHeap, n Neighbor) {
	heap.Push(h, n)
}

func popMin(h *minHeap) Neighbor {
	return heap.Pop(h).(Neighbor)
}

func pushMax(h *maxHeap, n Neighbor) {
	heap.Push(h, n)
}

func popMax(h *maxHeap) Neighbor {
	return heap.Pop(h).(Neighbor)
}

func peekMax(h *maxHeap) Neighbor {
	return (*h)[0]
}
