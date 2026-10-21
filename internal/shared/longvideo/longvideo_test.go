package longvideo

import (
	"math"
	"testing"
)

func TestWeights(t *testing.T) {
	d, r, s := Weights(256, 144, 1)
	if math.Abs(d+r+s-1) > 1e-9 || math.Abs(d-r) > 1e-9 {
		t.Fatalf("equal pixels: %v %v %v", d, r, s)
	}
	d, r, _ = Weights(1280, 720, 1)
	if d > 0.05 || r < 0.9 {
		t.Fatalf("720p: draft %v refine %v", d, r)
	}
	d2, _, _ := Weights(1280, 720, 4)
	if d2 <= d {
		t.Fatal("more devices must give the draft a larger share")
	}
}
