package hnsw

import "math/rand"

type DistanceFunc func(a, b []float32) float32

type Node struct {
	ID          uint32
	Vector      []float32
	Level       int
	Connections [][]uint32
}

type Neighbor struct {
	ID   uint32
	Dist float32
}

type Graph struct {
	Nodes          []Node
	EntryPoint     uint32
	MaxLevel       int
	M              int
	M0             int
	EfConstruction int
	ML             float64
	Distance       DistanceFunc
	UseHeuristic   bool
	Seed           int64
	Flat           bool
	rng            *rand.Rand
	visited        []uint64
	visitEpoch     uint64
}
