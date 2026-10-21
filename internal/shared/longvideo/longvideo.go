// Package longvideo holds the segment plan shared by the controller and the
// Nivra App (docs/architecture/LONG_VIDEO.md). It must match
// ai/nivra/inference/longvideo.py exactly.
package longvideo

import "math"

// SegmentFrames is the 1 + 8k frame count of one model clip: 8 s at 30 fps -> 241.
func SegmentFrames(seconds float64, fps int) int {
	k := int(math.Ceil(math.Round(seconds*float64(fps)*1e6)/1e6/8 - 1e-9))
	if k < 1 {
		k = 1
	}
	return 1 + 8*k
}

// TotalFrames is round(seconds * fps) + 1.
func TotalFrames(seconds float64, fps int) int {
	return int(math.Round(seconds*float64(fps))) + 1
}

// Plan returns each segment's [start, end] frames; neighbours share a frame.
func Plan(frames, segFrames int) [][2]int {
	step := segFrames - 1
	var out [][2]int
	for s := 0; s < frames-1; s += step {
		e := s + step
		if e > frames-1 {
			e = frames - 1
		}
		out = append(out, [2]int{s, e})
	}
	if len(out) == 0 {
		out = [][2]int{{0, frames - 1}}
	}
	return out
}

// Weights split a long video's overall progress between its stages by
// expected cost: the draft renders every frame at 256x144, the refine stage
// every frame at the target size spread over `parallel` devices, the stitch
// is cheap. (Fixed weights made the time-left estimate wildly pessimistic
// during the draft.)
func Weights(width, height, parallel int) (draft, refine, stitch float64) {
	if parallel < 1 {
		parallel = 1
	}
	dp := float64(256 * 144)
	tp := float64(width*height) / float64(parallel)
	if tp <= 0 {
		tp = 1280 * 720
	}
	stitch = 0.03
	draft = (1 - stitch) * dp / (dp + tp)
	refine = 1 - stitch - draft
	return
}
