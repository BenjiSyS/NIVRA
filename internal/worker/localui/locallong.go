package localui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"nivra/internal/shared/longvideo"
	"nivra/internal/shared/pyexec"
)

// Long videos with a model on this computer: draft -> segments rendered in
// parallel on every local GPU (one at a time on CPU) -> stitch, the same job
// types the controller hands to network workers.

// paramNum reads a numeric history parameter. The history holds the Go types
// the request was built with (int, int64, float64) until the app restarts and
// the file is reloaded (where everything is float64), so a plain float64
// assertion silently returned 0 for width, height, fps and seed.
func paramNum(p map[string]any, k string) float64 {
	switch v := p[k].(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case int32:
		return float64(v)
	case json.Number:
		f, _ := v.Float64()
		return f
	}
	return 0
}

type progressFunc func(f float64, stage string)

type fnReporter struct{ f progressFunc }

func (r fnReporter) Progress(f float64, s string)  { r.f(f, s) }
func (r fnReporter) Log(string, string)            {}
func (r fnReporter) Metric(string, float64, int64) {}

// runPy runs one executor job and returns its artifacts by kind.
func runPy(ctx context.Context, rt LocalRuntime, device, work, jobType, id string, params map[string]any,
	inputs map[string]any, onProgress progressFunc) (map[string]string, error) {
	if err := os.MkdirAll(work, 0o700); err != nil {
		return nil, err
	}
	pb, _ := json.Marshal(params)
	run := &pyexec.Runner{Interpreter: rt.Python, AIDir: rt.AIDir, FFmpeg: rt.FFmpeg, Threads: rt.Threads, Device: device}
	res, err := run.Run(ctx, work, pyexec.Job{JobType: jobType, JobID: id, AttemptID: id, Params: pb, Inputs: inputs}, fnReporter{onProgress})
	if err != nil {
		var f *pyexec.Failure
		if errors.As(err, &f) {
			return nil, errors.New(f.Message)
		}
		return nil, err
	}
	out := map[string]string{}
	for _, a := range res.Artifacts {
		p := a.Path
		if !filepath.IsAbs(p) {
			p = filepath.Join(work, p)
		}
		out[a.Kind] = p
	}
	return out, nil
}

// generateLocalLong renders a video longer than one model clip.
func (a *App) generateLocalLong(ctx context.Context, it histItem, m localModel, rt LocalRuntime, work string) (string, error) {
	inputs := map[string]any{"weights": filepath.Join(m.Path, "weights.safetensors"),
		"manifest": filepath.Join(m.Path, "model_version.json"), "vae_weights": filepath.Join(m.Path, "vae.safetensors")}
	num := func(k string) float64 { return paramNum(it.Params, k) }
	fps := int(num("fps"))
	if fps <= 0 {
		fps = m.DefaultFPS
	}
	seconds := num("seconds")
	common := map[string]any{"prompt": it.Prompt, "fps": fps}
	for _, k := range []string{"style", "sampler", "steps", "cfg_scale"} {
		if v, ok := it.Params[k]; ok {
			common[k] = v
		}
	}
	seed := int(num("seed"))
	// Output size: what the user asked for, else the model's native size.
	width, height := int(num("width")), int(num("height"))
	if width <= 0 || height <= 0 {
		width, height = m.DefaultWidth, m.DefaultHeight
	}
	segFrames := longvideo.SegmentFrames(m.DefaultSeconds, fps)
	plan := longvideo.Plan(longvideo.TotalFrames(seconds, fps), segFrames)
	with := func(extra map[string]any) map[string]any {
		p := map[string]any{}
		for k, v := range common {
			p[k] = v
		}
		for k, v := range extra {
			p[k] = v
		}
		return p
	}
	devices := rt.Devices
	if len(devices) == 0 {
		devices = []string{rt.Device}
	}
	var mu sync.Mutex
	segProg := make([]float64, len(plan))
	stageText := ""
	draftProg, stitchProg := 0.0, 0.0
	report := func() {
		mu.Lock()
		sum := 0.0
		done := 0
		for _, p := range segProg {
			sum += p
			if p >= 1 {
				done++
			}
		}
		wd, wr, ws := longvideo.Weights(width, height, len(devices))
		total := wd*draftProg + wr*sum/float64(len(plan)) + ws*stitchProg
		st := stageText
		if strings.HasPrefix(st, "rendering") {
			st = fmt.Sprintf("rendering segments: %d of %d done on %d device(s)", done, len(plan), len(devices))
		}
		mu.Unlock()
		a.update(it.ID, func(h *histItem) {
			if h.Status == "running" {
				h.Progress, h.Stage = total, st
			}
		})
	}
	stageText = "drawing the whole video at low resolution"
	draft, err := runPy(ctx, rt, devices[0], filepath.Join(work, "draft"), "ai.draft", it.ID+"-draft",
		with(map[string]any{"seed": seed, "seconds": seconds, "segment_frames": segFrames, "name": "draft"}), inputs,
		func(f float64, _ string) { mu.Lock(); draftProg = f; mu.Unlock(); report() })
	if err != nil {
		return "", fmt.Errorf("draft: %w", err)
	}
	draftVideo := draft["draft_video"]
	if draftVideo == "" {
		return "", errors.New("draft: no draft video")
	}
	mu.Lock()
	draftProg, stageText = 1, "rendering"
	mu.Unlock()
	report()

	// One goroutine per device pulls segments off a queue.
	segPaths := make([]string, len(plan))
	queue := make(chan int, len(plan))
	for i := range plan {
		queue <- i
	}
	close(queue)
	sctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	var firstErr error
	var errOnce sync.Once
	for _, dev := range devices {
		wg.Add(1)
		go func(dev string) {
			defer wg.Done()
			for i := range queue {
				if sctx.Err() != nil {
					return
				}
				seg := plan[i]
				in := map[string]any{"draft_video": draftVideo}
				for k, v := range inputs {
					in[k] = v
				}
				arts, err := runPy(sctx, rt, dev, filepath.Join(work, fmt.Sprintf("seg_%03d", i)), "ai.refine_segment",
					fmt.Sprintf("%s-seg%d", it.ID, i), with(map[string]any{"seed": (seed + i) % 2147483647,
						"width": width, "height": height, "index": i, "start": seg[0], "end": seg[1],
						"name": fmt.Sprintf("segment_%03d", i)}), in,
					func(f float64, _ string) { mu.Lock(); segProg[i] = f; mu.Unlock(); report() })
				if err == nil && arts["segment_video"] == "" {
					err = errors.New("no segment video")
				}
				if err != nil {
					errOnce.Do(func() { firstErr = fmt.Errorf("segment %d: %w", i, err); cancel() })
					return
				}
				mu.Lock()
				segPaths[i], segProg[i] = arts["segment_video"], 1
				mu.Unlock()
				report()
			}
		}(dev)
	}
	wg.Wait()
	if firstErr != nil {
		return "", firstErr
	}
	if ctx.Err() != nil {
		return "", ctx.Err()
	}

	mu.Lock()
	stageText = "joining the segments"
	mu.Unlock()
	report()
	in := map[string]any{}
	for i, p := range segPaths {
		in[fmt.Sprintf("segment_%03d", i)] = p
	}
	out, err := runPy(ctx, rt, "cpu", filepath.Join(work, "stitch"), "ai.stitch", it.ID+"-stitch",
		map[string]any{"fps": fps, "count": len(segPaths), "crossfade_frames": 4, "name": "video"}, in,
		func(f float64, _ string) { mu.Lock(); stitchProg = f; mu.Unlock(); report() })
	if err != nil {
		return "", fmt.Errorf("joining: %w", err)
	}
	if out["video"] == "" {
		return "", errors.New("joining: no video")
	}
	return out["video"], nil
}
