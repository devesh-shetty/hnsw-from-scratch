package datasets

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type GloVe struct {
	Words   []string
	Vectors [][]float32
	Dim     int
}

func LoadGloVe(path string, limit int) (*GloVe, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// Allow long lines for 300d embeddings.
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 10*1024*1024)

	capacity := 0
	if limit > 0 {
		capacity = limit
	}
	words := make([]string, 0, capacity)
	vectors := make([][]float32, 0, capacity)
	var dim int

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		if dim == 0 {
			dim = len(fields) - 1
		} else if len(fields)-1 != dim {
			return nil, fmt.Errorf("glove: inconsistent dim at line %d: got %d, expected %d", lineNum, len(fields)-1, dim)
		}

		v := make([]float32, dim)
		for i := 0; i < dim; i++ {
			val, err := strconv.ParseFloat(fields[i+1], 32)
			if err != nil {
				return nil, fmt.Errorf("glove: parse error at line %d col %d: %w", lineNum, i+1, err)
			}
			v[i] = float32(val)
		}

		words = append(words, fields[0])
		vectors = append(vectors, v)
		if limit > 0 && len(vectors) >= limit {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return &GloVe{Words: words, Vectors: vectors, Dim: dim}, nil
}
