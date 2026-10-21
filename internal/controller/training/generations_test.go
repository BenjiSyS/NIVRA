package training

import "testing"

func TestSegmentPlan(t *testing.T) {
	if got := segmentFrames(8, 30); got != 241 {
		t.Fatalf("segmentFrames(8,30)=%d", got)
	}
	if got := segmentFrames(8, 24); got != 193 {
		t.Fatalf("segmentFrames(8,24)=%d", got)
	}
	// 20 s at 30 fps = 601 frames -> segments of 240 frame steps sharing boundaries.
	plan := segmentPlan(601, 241)
	want := [][2]int{{0, 240}, {240, 480}, {480, 600}}
	if len(plan) != len(want) {
		t.Fatalf("plan %v", plan)
	}
	for i := range want {
		if plan[i] != want[i] {
			t.Fatalf("plan %v want %v", plan, want)
		}
	}
	// Exactly one clip.
	if p := segmentPlan(241, 241); len(p) != 1 || p[0] != [2]int{0, 240} {
		t.Fatalf("single %v", p)
	}
	// One hour at 30 fps.
	if n := len(segmentPlan(3600*30+1, 241)); n != 450 {
		t.Fatalf("1h segments %d", n)
	}
	// Every frame covered, neighbours share exactly one frame.
	p := segmentPlan(1000, 193)
	if p[0][0] != 0 || p[len(p)-1][1] != 999 {
		t.Fatalf("coverage %v", p)
	}
	for i := 1; i < len(p); i++ {
		if p[i][0] != p[i-1][1] {
			t.Fatalf("gap at %d: %v", i, p)
		}
	}
}
