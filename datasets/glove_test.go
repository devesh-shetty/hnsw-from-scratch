package datasets

import "testing"

func TestLoadGloVe(t *testing.T) {
	g, err := LoadGloVe("testdata/glove.txt", 0)
	if err != nil {
		t.Fatalf("LoadGloVe error: %v", err)
	}
	if g.Dim != 3 {
		t.Fatalf("expected dim 3, got %d", g.Dim)
	}
	if len(g.Words) != 2 {
		t.Fatalf("expected 2 words, got %d", len(g.Words))
	}
	if g.Words[0] != "king" {
		t.Fatalf("expected first word 'king', got %q", g.Words[0])
	}
}
