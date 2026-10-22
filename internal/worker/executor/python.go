package executor

import (
	"context"
	"errors"
	"strconv"
	"strings"

	"nivra/internal/shared/pyexec"
)

// Python runs AI job types through the shared pyexec runner.
type Python struct {
	Interpreter string
	AIDir       string
	FFmpeg      string // configured ffmpeg path ("" = auto)
	JobTypes    []string
}

// Types implements Executor.
func (p *Python) Types() []string { return p.JobTypes }

type reporterAdapter struct{ r Reporter }

func (a reporterAdapter) Progress(f float64, s string)  { a.r.Progress(f, s) }
func (a reporterAdapter) Log(l, m string)               { a.r.Log(l, m) }
func (a reporterAdapter) Metric(string, float64, int64) {}

// Run implements Executor.
func (p *Python) Run(ctx context.Context, t *Task, r Reporter) (*Output, error) {
	device := "cpu"
	if t.GPUIndex >= 0 {
		device = "cuda:" + strconv.Itoa(t.GPUIndex)
	}
	inputs := map[string]any{}
	for k, v := range t.Inputs {
		inputs[k] = v
	}
	run := &pyexec.Runner{Interpreter: p.Interpreter, AIDir: p.AIDir, FFmpeg: p.FFmpeg, Threads: t.Threads, Device: device}
	res, err := run.Run(ctx, t.WorkDir, pyexec.Job{JobType: t.Assignment.JobType, JobID: t.Assignment.JobID,
		AttemptID: t.Assignment.AttemptID, Params: t.Assignment.Params, Inputs: inputs,
		Limits: map[string]int64{"max_ram_mb": t.RAMMB, "max_vram_mb": t.VRAMMB}}, reporterAdapter{r})
	if errors.Is(err, pyexec.ErrStopped) {
		return nil, ErrStopped
	}
	var f *pyexec.Failure
	if errors.As(err, &f) {
		return nil, &Failure{Code: f.Code, Message: f.Message, Retryable: f.Retryable}
	}
	if err != nil {
		return nil, err
	}
	o := &Output{Result: res.Summary}
	for _, a := range res.Artifacts {
		ct := "application/octet-stream"
		switch {
		case strings.HasSuffix(a.Path, ".json"):
			ct = "application/json"
		case strings.HasSuffix(a.Path, ".mp4"):
			ct = "video/mp4"
		}
		o.Artifacts = append(o.Artifacts, OutputArtifact{Kind: a.Kind, Path: a.Path, ContentType: ct})
	}
	return o, nil
}
