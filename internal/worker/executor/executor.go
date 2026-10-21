// Package executor runs job attempts: Go-native builtin jobs in-process and
// AI jobs in an isolated Python subprocess speaking nivra.exec/1.
package executor

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"nivra/internal/shared/builtin"
	"nivra/internal/shared/protocol"
)

// ErrStopped means the task was stopped on request (it will be requeued).
var ErrStopped = errors.New("task stopped")

// Failure is an execution failure with a protocol error code.
type Failure struct {
	Code      string
	Message   string
	Retryable bool
}

func (f *Failure) Error() string { return f.Code + ": " + f.Message }

// Task is one attempt to execute.
type Task struct {
	Assignment protocol.JobAssignment
	WorkDir    string
	// Inputs maps input role (or artifact id when role is empty) to a local file.
	Inputs map[string]string
	// Threads bounds CPU threads; GPUIndex < 0 means CPU only.
	Threads  int
	GPUIndex int
	RAMMB    int64
	VRAMMB   int64
}

// Reporter receives progress and log lines from a running task.
type Reporter interface {
	Progress(fraction float64, stage string)
	Log(level, msg string)
}

// Output is the result of a successful task.
type Output struct {
	Result json.RawMessage
	// Artifacts are uploaded and attached to the result, keyed by kind.
	Artifacts []OutputArtifact
}

// OutputArtifact is one produced file.
type OutputArtifact struct {
	Kind        string
	Path        string
	ContentType string
}

// Executor runs a family of job types.
type Executor interface {
	Types() []string
	Run(ctx context.Context, t *Task, r Reporter) (*Output, error)
}

// Builtin runs Go-native job types.
type Builtin struct{}

// Types implements Executor.
func (Builtin) Types() []string { return []string{builtin.TypeSHA256Chain, builtin.TypeSleep} }

// Run implements Executor.
func (Builtin) Run(ctx context.Context, t *Task, r Reporter) (*Output, error) {
	switch t.Assignment.JobType {
	case builtin.TypeSHA256Chain:
		var p builtin.SHA256ChainParams
		if err := builtin.Decode(t.Assignment.Params, &p); err != nil {
			return nil, &Failure{Code: "INVALID_SPEC", Message: err.Error()}
		}
		r.Progress(0, "hashing")
		d, err := builtin.SHA256Chain(ctx, p, func(f float64) { r.Progress(f, "hashing") })
		if err != nil {
			if ctx.Err() != nil {
				return nil, ErrStopped
			}
			return nil, &Failure{Code: "INVALID_SPEC", Message: err.Error()}
		}
		b, _ := json.Marshal(builtin.SHA256ChainResult{Digest: d})
		return &Output{Result: b}, nil
	case builtin.TypeSleep:
		var p builtin.SleepParams
		if err := builtin.Decode(t.Assignment.Params, &p); err != nil {
			return nil, &Failure{Code: "INVALID_SPEC", Message: err.Error()}
		}
		start := time.Now()
		total := time.Duration(p.Seconds * float64(time.Second))
		tick := time.NewTicker(250 * time.Millisecond)
		defer tick.Stop()
		for {
			el := time.Since(start)
			frac := 1.0
			if total > 0 {
				frac = min(float64(el)/float64(total), 1)
			}
			if p.FailAt > 0 && frac >= p.FailAt {
				return nil, &Failure{Code: "INTERNAL", Message: "injected failure (fail_at)", Retryable: true}
			}
			if el >= total {
				break
			}
			r.Progress(frac, "sleeping")
			select {
			case <-ctx.Done():
				return nil, ErrStopped
			case <-tick.C:
			}
		}
		b, _ := json.Marshal(builtin.SleepResult{SleptMS: time.Since(start).Milliseconds()})
		return &Output{Result: b}, nil
	}
	return nil, &Failure{Code: "INVALID_SPEC", Message: "unsupported job type " + t.Assignment.JobType}
}
