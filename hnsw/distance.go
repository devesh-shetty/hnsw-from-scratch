package hnsw

import "math"

func L2Squared(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func Dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// CosineDistance returns 1 - cosine similarity. Requires pre-normalized vectors
// for cosine to be meaningful.
func CosineDistance(a, b []float32) float32 {
	return 1 - Dot(a, b)
}

func Normalize(v []float32) {
	var sum float64
	for i := range v {
		sum += float64(v[i] * v[i])
	}
	n := float32(math.Sqrt(sum))
	if n == 0 {
		return
	}
	for i := range v {
		v[i] /= n
	}
}
