package training

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"nivra/internal/shared/pyexec"
	"nivra/internal/shared/secure"
)

// logReporter forwards Python output to the controller log.
type logReporter struct {
	s   *Service
	job string
	run string
}

func (r logReporter) Progress(float64, string) {}
func (r logReporter) Log(level, msg string) {
	switch level {
	case "error":
		r.s.d.Log.Error(msg, "service", "python", "job_type", r.job, "run_id", r.run)
	case "warn":
		r.s.d.Log.Warn(msg, "service", "python", "job_type", r.job, "run_id", r.run)
	case "debug":
	default:
		r.s.d.Log.Debug(msg, "job_type", r.job)
	}
}
func (r logReporter) Metric(string, float64, int64) {}

// pyMu serialises controller-side Python jobs: the Pi has 4 cores and
// shares them with unrelated services.
var pyMu sync.Mutex

// runPython executes a controller-side job in a fresh temporary directory.
// The caller must call cleanup when done with the outputs.
func (s *Service) runPython(ctx context.Context, jobType, runID string, params any, inputs map[string]any) (*pyexec.Result, string, func(), error) {
	pyMu.Lock()
	defer pyMu.Unlock()
	dir := filepath.Join(s.d.Cfg.DataDir, "tmp", "py-"+secure.RandomToken()[:16])
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, "", func() {}, err
	}
	cleanup := func() { _ = os.RemoveAll(dir) }
	pb, err := json.Marshal(params)
	if err != nil {
		cleanup()
		return nil, "", func() {}, err
	}
	r := &pyexec.Runner{Interpreter: s.d.Cfg.PythonPath, AIDir: s.d.Cfg.AIPackageDir, Threads: 2, Device: "cpu"}
	res, err := r.Run(ctx, dir, pyexec.Job{JobType: jobType, JobID: "controller", AttemptID: "controller", Params: pb, Inputs: inputs},
		logReporter{s: s, job: jobType, run: runID})
	if err != nil {
		cleanup()
		return nil, "", func() {}, fmt.Errorf("%s: %w", jobType, err)
	}
	return res, dir, cleanup, nil
}

// artifactByKind finds an output artifact.
func artifactByKind(res *pyexec.Result, kind string) (pyexec.Artifact, bool) {
	for _, a := range res.Artifacts {
		if a.Kind == kind {
			return a, true
		}
	}
	return pyexec.Artifact{}, false
}
