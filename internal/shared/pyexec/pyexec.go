// Package pyexec runs `python -m nivra.cli exec` subprocesses speaking the
// nivra.exec/1 protocol (docs/architecture/EXECUTOR_PROTOCOL.md). It is used
// by workers for AI jobs and by the controller for update validation and
// aggregation.
package pyexec

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Protocol is the executor protocol version string.
const Protocol = "nivra.exec/1"

// Exit codes defined by the protocol.
const (
	ExitOK           = 0
	ExitInternal     = 1
	ExitInvalidSpec  = 2
	ExitStopped      = 3
	ExitOOM          = 4
	ExitInputInvalid = 5
)

// ErrStopped means the job stopped because the context was cancelled.
var ErrStopped = errors.New("python job stopped")

// Failure is a non-successful execution.
type Failure struct {
	ExitCode  int
	Code      string
	Message   string
	Retryable bool
}

func (f *Failure) Error() string {
	return fmt.Sprintf("%s (exit %d): %s", f.Code, f.ExitCode, f.Message)
}

// Reporter receives progress, metrics and logs.
type Reporter interface {
	Progress(fraction float64, stage string)
	Log(level, msg string)
	Metric(name string, value float64, step int64)
}

// Artifact is a file produced by the job inside its output directory.
type Artifact struct {
	Kind   string `json:"kind"`
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// Result is a successful execution.
type Result struct {
	Summary   json.RawMessage
	Artifacts []Artifact
}

// Job is the job file content.
type Job struct {
	JobType   string
	JobID     string
	AttemptID string
	Params    json.RawMessage
	Inputs    map[string]any
	Limits    map[string]int64
}

// Runner launches executor subprocesses.
type Runner struct {
	Interpreter string
	AIDir       string
	Threads     int
	Device      string // cpu | cuda | cuda:N
	StopGrace   time.Duration
	// FFmpeg is the configured ffmpeg file or folder ("" = find it on PATH or in the
	// winget folder); the resolved paths are passed as NIVRA_FFMPEG / NIVRA_FFPROBE.
	FFmpeg string
	// StallTimeout kills the process tree and fails retryably (code STALLED) when no
	// output line arrived for this long. 0 = the default for the job type
	// (StallTimeoutFor); negative = no watchdog.
	StallTimeout time.Duration
	// StateDir is persistent per-worker state kept across jobs (e.g. the int8
	// error-feedback residual of HD training); passed as NIVRA_STATE_DIR.
	StateDir string

	// stallCheck overrides how often the watchdog looks (tests).
	stallCheck time.Duration
}

type jobFile struct {
	Protocol  string           `json:"protocol"`
	JobType   string           `json:"job_type"`
	JobID     string           `json:"job_id"`
	AttemptID string           `json:"attempt_id"`
	Params    json.RawMessage  `json:"params"`
	Inputs    map[string]any   `json:"inputs"`
	OutputDir string           `json:"output_dir"`
	StopFile  string           `json:"stop_file"`
	Limits    map[string]int64 `json:"limits"`
}

type event struct {
	Event    string          `json:"event"`
	Level    string          `json:"level"`
	Message  string          `json:"message"`
	Stage    string          `json:"stage"`
	Fraction float64         `json:"fraction"`
	Name     string          `json:"name"`
	Value    float64         `json:"value"`
	Step     int64           `json:"step"`
	Kind     string          `json:"kind"`
	Path     string          `json:"path"`
	SHA256   string          `json:"sha256"`
	Bytes    int64           `json:"bytes"`
	Summary  json.RawMessage `json:"summary"`
	Code     string          `json:"code"`
}

// Run executes job in workDir. Cancelling ctx requests a graceful stop via
// the stop file; after StopGrace the whole process tree is killed.
func (r *Runner) Run(ctx context.Context, workDir string, job Job, rep Reporter) (*Result, error) {
	out := filepath.Join(workDir, "out")
	if err := os.MkdirAll(out, 0o750); err != nil {
		return nil, err
	}
	jf := jobFile{Protocol: Protocol, JobType: job.JobType, JobID: job.JobID, AttemptID: job.AttemptID, Params: job.Params,
		Inputs: job.Inputs, OutputDir: out, StopFile: filepath.Join(workDir, "STOP"), Limits: job.Limits}
	if jf.Inputs == nil {
		jf.Inputs = map[string]any{}
	}
	if jf.Limits == nil {
		jf.Limits = map[string]int64{}
	}
	if len(jf.Params) == 0 {
		jf.Params = json.RawMessage(`{}`)
	}
	_ = os.Remove(jf.StopFile)
	b, _ := json.MarshalIndent(jf, "", "  ")
	jobPath := filepath.Join(workDir, "job.json")
	if err := os.WriteFile(jobPath, b, 0o640); err != nil {
		return nil, err
	}
	interp := r.Interpreter
	if interp == "" {
		interp = "python"
	}
	cmd := exec.Command(interp, "-m", "nivra.cli", "exec", "--job", jobPath)
	cmd.Dir = workDir
	threads := r.Threads
	if threads <= 0 {
		threads = 2
	}
	device := r.Device
	if device == "" {
		device = "cpu"
	}
	env := os.Environ()
	if r.AIDir != "" {
		abs, _ := filepath.Abs(r.AIDir)
		env = append(env, "PYTHONPATH="+abs)
	}
	env = append(env, "NIVRA_THREADS="+strconv.Itoa(threads), "NIVRA_DEVICE="+device, "PYTHONUNBUFFERED=1",
		"OMP_NUM_THREADS="+strconv.Itoa(threads), "PYTHONIOENCODING=utf-8", "PYTHONUTF8=1")
	env = append(env, ResolveFFmpeg(r.FFmpeg).Env()...)
	if r.StateDir != "" {
		if abs, err := filepath.Abs(r.StateDir); err == nil {
			env = append(env, "NIVRA_STATE_DIR="+abs)
		}
	}
	cmd.Env = env
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderrPath := filepath.Join(workDir, "stderr.log")
	stderrF, err := os.Create(stderrPath)
	if err != nil {
		return nil, err
	}
	defer stderrF.Close()
	cmd.Stderr = stderrF
	pg, err := startGroup(cmd)
	if err != nil {
		return nil, &Failure{ExitCode: -1, Code: "INTERNAL", Message: "cannot start python: " + err.Error()}
	}
	defer pg.Close()

	grace := r.StopGrace
	if grace <= 0 {
		grace = 30 * time.Second
	}
	done := make(chan struct{})
	var mu sync.Mutex
	stopped := false
	stalled := false
	var lastOut atomic.Int64
	lastOut.Store(time.Now().UnixNano())
	stall := r.StallTimeout
	if stall == 0 {
		stall = StallTimeoutFor(job.JobType)
	}
	if stall > 0 {
		check := r.stallCheck
		if check <= 0 {
			check = min(stall/4, 10*time.Second)
		}
		go func() {
			tick := time.NewTicker(check)
			defer tick.Stop()
			for {
				select {
				case <-done:
					return
				case <-tick.C:
				}
				if time.Since(time.Unix(0, lastOut.Load())) < stall {
					continue
				}
				mu.Lock()
				stalled = true
				mu.Unlock()
				if rep != nil {
					rep.Log("error", fmt.Sprintf("python job printed nothing for %s; killing its process tree", stall.Round(time.Second)))
				}
				_ = pg.Kill()
				return
			}
		}()
	}
	go func() {
		select {
		case <-done:
			return
		case <-ctx.Done():
		}
		mu.Lock()
		stopped = true
		mu.Unlock()
		_ = os.WriteFile(jf.StopFile, []byte("stop"), 0o640)
		select {
		case <-done:
		case <-time.After(grace):
			if rep != nil {
				rep.Log("warn", "python job did not stop within the grace period; killing its process tree")
			}
			_ = pg.Kill()
		}
	}()

	var res *event
	var errEv *event
	var arts []Artifact
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 64<<10), 8<<20)
	for sc.Scan() {
		lastOut.Store(time.Now().UnixNano())
		line := sc.Bytes()
		var ev event
		if len(line) == 0 || line[0] != '{' || json.Unmarshal(line, &ev) != nil {
			if rep != nil && len(line) > 0 {
				rep.Log("info", truncate(string(line), 2000))
			}
			continue
		}
		switch ev.Event {
		case "log":
			if rep != nil {
				rep.Log(ev.Level, truncate(ev.Message, 2000))
			}
		case "progress":
			if rep != nil {
				rep.Progress(min(max(ev.Fraction, 0), 1), truncate(ev.Stage, 100))
			}
		case "metric":
			if rep != nil {
				rep.Metric(ev.Name, ev.Value, ev.Step)
			}
		case "artifact":
			p := ev.Path
			if !filepath.IsAbs(p) {
				p = filepath.Join(out, p)
			}
			rel, err := filepath.Rel(out, p)
			if err != nil || strings.HasPrefix(rel, "..") {
				if rep != nil {
					rep.Log("error", "ignoring artifact outside the output directory: "+ev.Path)
				}
				continue
			}
			arts = append(arts, Artifact{Kind: ev.Kind, Path: p, SHA256: ev.SHA256, Bytes: ev.Bytes})
		case "result":
			e := ev
			res = &e
		case "error":
			e := ev
			errEv = &e
		}
	}
	if err := sc.Err(); err != nil {
		// A line longer than the buffer (or a read error) ends the scanner; keep
		// draining so Python never blocks on a full pipe.
		if rep != nil {
			rep.Log("warn", "python output could not be read further ("+err.Error()+"); ignoring the rest")
		}
		_, _ = io.Copy(io.Discard, stdout)
	}
	werr := cmd.Wait()
	close(done)
	code := 0
	if werr != nil {
		var ee *exec.ExitError
		if errors.As(werr, &ee) {
			code = ee.ExitCode()
		} else {
			code = -1
		}
	}
	mu.Lock()
	wasStopped, wasStalled := stopped, stalled
	mu.Unlock()
	if wasStalled && !wasStopped {
		return nil, &Failure{ExitCode: code, Code: CodeStalled, Retryable: true,
			Message: "the task made no progress for " + humanDur(stall) + " and was stopped"}
	}
	if wasStopped && code != 0 {
		return nil, ErrStopped
	}
	if code != 0 || res == nil {
		f := &Failure{ExitCode: code, Code: "INTERNAL", Message: fmt.Sprintf("python exited with code %d", code), Retryable: true}
		if errEv != nil {
			f.Code, f.Message = errEv.Code, errEv.Message
		} else if tail := tailFile(stderrPath, 1500); tail != "" {
			f.Message += ": " + tail
		}
		switch {
		case code == ExitOOM || f.Code == "OOM":
			f.Code, f.Retryable = "OOM", true
		case code == ExitInvalidSpec || code == ExitInputInvalid || f.Code == "INVALID_SPEC" || f.Code == "INPUT_INVALID":
			f.Retryable = false
		}
		f.Message = truncate(f.Message, 2000)
		return nil, f
	}
	summary := res.Summary
	if len(summary) == 0 {
		summary = json.RawMessage(`{}`)
	}
	return &Result{Summary: summary, Artifacts: arts}, nil
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func tailFile(path string, n int64) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return ""
	}
	off := max(st.Size()-n, 0)
	b := make([]byte, st.Size()-off)
	_, _ = f.ReadAt(b, off)
	return strings.TrimSpace(string(b))
}

func humanDur(d time.Duration) string {
	if d >= time.Minute {
		return fmt.Sprintf("%d minutes", int(d.Round(time.Minute)/time.Minute))
	}
	return fmt.Sprintf("%d seconds", int(d.Round(time.Second)/time.Second))
}
