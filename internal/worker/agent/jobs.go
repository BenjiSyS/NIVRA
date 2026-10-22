package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"nivra/internal/shared/config"
	"nivra/internal/shared/protocol"
	"nivra/internal/shared/secure"
	"nivra/internal/worker/client"
	"nivra/internal/worker/executor"
	"nivra/internal/worker/keepawake"
)

type stopReason int

const (
	stopNone stopReason = iota
	stopRequeue
	stopDiscard
)

// activeTask is the attempt currently executing on this worker.
type activeTask struct {
	a      protocol.JobAssignment
	cancel context.CancelFunc
	done   chan struct{}

	mu       sync.Mutex
	reason   stopReason
	phase    string // downloading | running | uploading | submitting
	stage    string
	progress float64
	started  time.Time
	upload   string
}

// TaskView is the Worker website's view of the current task.
type TaskView struct {
	AttemptID string    `json:"attempt_id"`
	JobID     string    `json:"job_id"`
	JobType   string    `json:"job_type"`
	Phase     string    `json:"phase"`
	Stage     string    `json:"stage"`
	Progress  float64   `json:"progress"`
	StartedAt time.Time `json:"started_at"`
	ElapsedS  float64   `json:"elapsed_sec"`
	Upload    string    `json:"upload"`
	Stopping  bool      `json:"stopping"`
}

func (t *activeTask) view() TaskView {
	t.mu.Lock()
	defer t.mu.Unlock()
	return TaskView{AttemptID: t.a.AttemptID, JobID: t.a.JobID, JobType: t.a.JobType, Phase: t.phase, Stage: t.stage,
		Progress: t.progress, StartedAt: t.started, ElapsedS: time.Since(t.started).Seconds(), Upload: t.upload,
		Stopping: t.reason != stopNone}
}

func (t *activeTask) status() protocol.AttemptStatus {
	t.mu.Lock()
	defer t.mu.Unlock()
	return protocol.AttemptStatus{AttemptID: t.a.AttemptID, FencingToken: t.a.FencingToken, Progress: t.progress, Stage: t.stage}
}

func (t *activeTask) stop(r stopReason) {
	t.mu.Lock()
	if t.reason == stopNone || r == stopDiscard {
		t.reason = r
	}
	t.mu.Unlock()
	t.cancel()
}

func (t *activeTask) set(phase, stage string, progress float64) {
	t.mu.Lock()
	if phase != "" {
		t.phase = phase
	}
	if stage != "" {
		t.stage = stage
	}
	if progress >= 0 {
		t.progress = progress
	}
	t.mu.Unlock()
}

// workLoop leases and starts jobs. Tasks run on the agent context, so a
// brief controller outage does not kill them; results wait in the outbox.
func (a *Agent) workLoop(ctx context.Context, cl *client.Client) error {
	failures := 0
	for {
		if t := a.currentTask(); t != nil {
			select {
			case <-t.done:
				continue
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		if paused, _ := a.effectivelyPaused(); paused {
			a.refreshState()
			if !sleepCtx(ctx, 3*time.Second) {
				return ctx.Err()
			}
			continue
		}
		a.refreshState()
		job, err := cl.NextJob(ctx, 25*time.Second)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if isRevoked(err) {
				a.revoked(err)
				return errRevoked
			}
			failures++
			a.logf("warn", "jobs", "job poll failed: %v", err)
			if failures >= 5 {
				return fmt.Errorf("job polling failed repeatedly: %w", err)
			}
			if !sleepCtx(ctx, jitter(5*time.Second)) {
				return ctx.Err()
			}
			continue
		}
		failures = 0
		if job == nil {
			continue
		}
		a.startTask(cl, job)
	}
}

func (a *Agent) startTask(cl *client.Client, job *protocol.JobAssignment) {
	if err := a.validateAssignment(job); err != nil {
		go a.rejectAssignment(cl, job, err)
		return
	}
	a.gate.mu.Lock() // released below, after a.task is set
	a.mu.Lock()
	loopCtx := context.Background()
	if a.loopCancel != nil {
		loopCtx = a.loopCtxLocked()
	}
	ctx, cancel := context.WithCancel(loopCtx)
	t := &activeTask{a: *job, cancel: cancel, done: make(chan struct{}), phase: "starting", started: time.Now()}
	a.task = t
	conflict := a.gate.LocalActive()
	a.mu.Unlock()
	a.gate.mu.Unlock()
	if conflict {
		// A local generation started while this lease was in flight: hand it back.
		t.stop(stopRequeue)
	}
	a.logf("info", "jobs", "leased %s attempt %s (job %s, attempt #%d)", job.JobType, job.AttemptID, job.JobID, job.AttemptNumber)
	go a.runTask(ctx, cl, t)
}

// rejectAssignment reports a malformed assignment as a non-retryable failure. If
// even the attempt id is unusable nothing can be reported; the lease just expires.
func (a *Agent) rejectAssignment(cl *client.Client, job *protocol.JobAssignment, why error) {
	a.logf("error", "jobs", "rejected assignment: %v", why)
	a.mu.Lock()
	a.failed++
	a.mu.Unlock()
	if !isUUID(job.AttemptID) || !isUUID(cl.Identity().ProjectID) {
		return
	}
	entry := outboxEntry{ProjectID: cl.Identity().ProjectID, AttemptID: job.AttemptID, JobType: "", Created: time.Now(),
		Fail: &protocol.JobFailRequest{FencingToken: job.FencingToken, Code: "INVALID_SPEC",
			Message: "the worker rejected this job: " + why.Error(), Retryable: false}}
	if err := a.writeOutbox(entry); err != nil {
		a.logf("error", "jobs", "cannot persist rejection for %s: %v", job.AttemptID, err)
	}
	a.submitOutbox(context.Background(), cl, entry)
}

// runTask executes one attempt end to end.
func (a *Agent) runTask(ctx context.Context, cl *client.Client, t *activeTask) {
	defer func() {
		t.cancel()
		a.mu.Lock()
		if a.task == t {
			a.task = nil
		}
		a.mu.Unlock()
		close(t.done)
		a.refreshState()
	}()
	a.mu.Lock()
	sleepOK := a.settings.AllowSleepOnBattery
	a.mu.Unlock()
	defer keepawake.Acquire(sleepOK)() // no sleeping mid-task
	job := t.a
	bg := func() (context.Context, context.CancelFunc) {
		return context.WithTimeout(context.Background(), 30*time.Second)
	}

	actx, cancel := bg()
	err := cl.Ack(actx, &job)
	cancel()
	if err != nil {
		if protocol.IsCode(err, protocol.ErrStaleLease) || protocol.IsCode(err, protocol.ErrNotFound) {
			a.logf("warn", "jobs", "attempt %s no longer valid at ack; skipping", job.AttemptID)
			return
		}
		a.logf("warn", "jobs", "ack failed (%v); continuing, the heartbeat will reconcile", err)
	}

	// Progress publisher: at most one progress call every 2 s.
	pubDone := make(chan struct{})
	go func() {
		defer close(pubDone)
		tick := time.NewTicker(2 * time.Second)
		defer tick.Stop()
		var last float64 = -1
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
			}
			v := t.view()
			if v.Progress == last && v.Phase != "uploading" {
				continue
			}
			last = v.Progress
			state := "running"
			if v.Phase == "uploading" {
				state = "uploading"
			}
			pctx, cancel := context.WithTimeout(ctx, 15*time.Second)
			err := cl.Progress(pctx, job.AttemptID, protocol.JobProgressRequest{FencingToken: job.FencingToken, State: state,
				Progress: v.Progress, Stage: v.Stage})
			cancel()
			if protocol.IsCode(err, protocol.ErrStaleLease) {
				a.logf("warn", "jobs", "controller says attempt %s is stale; stopping", job.AttemptID)
				t.stop(stopDiscard)
				return
			}
		}
	}()

	out, err := a.execute(ctx, cl, t)
	// Upload artifacts while still holding the lease.
	var artifactID string
	artifactIDs := map[string]string{}
	if err == nil && len(out.Artifacts) > 16 {
		err = &executor.Failure{Code: "INTERNAL", Message: "too many artifacts", Retryable: false}
	}
	for i := 0; err == nil && i < len(out.Artifacts); i++ {
		art := out.Artifacts[i]
		t.set("uploading", fmt.Sprintf("uploading %s (%d/%d)", art.Kind, i+1, len(out.Artifacts)), -1)
		var id string
		id, err = cl.Upload(ctx, job.AttemptID, job.FencingToken, art.Path, art.Kind, art.ContentType, func(sent, total int64) {
			t.mu.Lock()
			t.upload = fmt.Sprintf("%s: %d/%d bytes", art.Kind, sent, total)
			t.mu.Unlock()
		})
		if err != nil {
			if ctx.Err() == nil {
				err = &executor.Failure{Code: "UPLOAD_FAILED", Message: err.Error(), Retryable: true}
			}
			break
		}
		if artifactID == "" {
			artifactID = id
		}
		artifactIDs[art.Kind] = id
	}
	t.mu.Lock()
	reason := t.reason
	t.mu.Unlock()
	// Was the task stopped from outside? Must be read BEFORE t.cancel(): that call
	// (made only to end the progress publisher) cancels this same context, so
	// reading it afterwards made every failure look like a stop and an invalid
	// request (e.g. an unknown style) was requeued forever instead of failing.
	interrupted := ctx.Err() != nil
	t.cancel()
	<-pubDone

	entry := outboxEntry{ProjectID: cl.Identity().ProjectID, AttemptID: job.AttemptID, JobType: job.JobType, Created: time.Now()}
	switch {
	case reason == stopDiscard:
		a.logf("info", "jobs", "attempt %s discarded", job.AttemptID)
		a.cleanupWork(job.AttemptID, false)
		return
	case errors.Is(err, executor.ErrStopped) || (err != nil && interrupted):
		a.logf("info", "jobs", "attempt %s stopped; returning it to the queue", job.AttemptID)
		entry.Fail = &protocol.JobFailRequest{FencingToken: job.FencingToken, Code: "STOPPED", Message: "stopped on request", Retryable: true}
	case err != nil:
		var f *executor.Failure
		if !errors.As(err, &f) {
			f = &executor.Failure{Code: "INTERNAL", Message: err.Error(), Retryable: true}
		}
		a.logf("error", "jobs", "attempt %s failed: %s: %s", job.AttemptID, f.Code, f.Message)
		entry.Fail = &protocol.JobFailRequest{FencingToken: job.FencingToken, Code: f.Code, Message: f.Message, Retryable: f.Retryable}
		a.mu.Lock()
		a.failed++
		a.mu.Unlock()
	default:
		t.set("submitting", "submitting result", 1)
		entry.Complete = &protocol.JobCompleteRequest{FencingToken: job.FencingToken, Result: out.Result, ArtifactID: artifactID, Artifacts: artifactIDs,
			Checksum: secure.SHA256Hex(out.Result), DurationMS: time.Since(t.started).Milliseconds()}
	}
	if err := a.writeOutbox(entry); err != nil {
		a.logf("error", "jobs", "cannot persist result for %s: %v", job.AttemptID, err)
	}
	a.submitOutbox(context.Background(), cl, entry)
	a.cleanupWork(job.AttemptID, entry.Fail != nil && entry.Fail.Code != "STOPPED")
}

// execute downloads inputs and runs the executor.
func (a *Agent) execute(ctx context.Context, cl *client.Client, t *activeTask) (*executor.Output, error) {
	job := t.a
	a.mu.Lock()
	ex, ok := a.executors[job.JobType]
	s := a.settings
	gpuOK := a.hw.AI.CUDA && s.GPUIndex >= 0 && s.GPUIndex < len(a.hw.GPUs)
	a.mu.Unlock()
	if !ok {
		return nil, &executor.Failure{Code: "UNSUPPORTED", Message: "this worker has no executor for " + job.JobType, Retryable: true}
	}
	if ctx.Err() != nil {
		return nil, executor.ErrStopped
	}
	project := cl.Identity().ProjectID
	cacheDir, err := a.safeJoin("cache", project)
	if err != nil || !isUUID(project) {
		return nil, &executor.Failure{Code: "INVALID_SPEC", Message: "unsafe project id", Retryable: false}
	}
	if err := os.MkdirAll(cacheDir, 0o700); err != nil {
		return nil, err
	}
	inputs, err := a.fetchInputs(ctx, t, job.Inputs, cacheDir, s.DiskCacheMB<<20, cl.Download)
	if err != nil {
		return nil, err
	}
	work, err := a.safeJoin("work", job.AttemptID)
	if err != nil {
		return nil, &executor.Failure{Code: "INVALID_SPEC", Message: "unsafe attempt id", Retryable: false}
	}
	if err := os.MkdirAll(work, 0o700); err != nil {
		return nil, err
	}
	gi := -1
	if gpuOK {
		gi = s.GPUIndex
	}
	t.set("running", "starting", 0)
	return ex.Run(ctx, &executor.Task{Assignment: job, WorkDir: work, Inputs: inputs, Threads: s.CPUThreads, GPUIndex: gi,
		RAMMB: s.RAMBudgetMB, VRAMMB: s.VRAMBudgetMB}, taskReporter{a: a, t: t})
}

type taskReporter struct {
	a *Agent
	t *activeTask
}

func (r taskReporter) Progress(f float64, stage string) { r.t.set("running", stage, f) }
func (r taskReporter) Log(level, msg string) {
	if level == "" {
		level = "info"
	}
	r.a.logf(level, "task", "%s", msg)
}

func (a *Agent) cleanupWork(attemptID string, keep bool) {
	dir, err := a.safeJoin("work", attemptID)
	if err != nil || !isUUID(attemptID) {
		a.logf("warn", "jobs", "not removing work directory for unsafe attempt id %q", short(attemptID))
		return
	}
	if keep {
		a.logf("info", "jobs", "kept failed work directory for diagnostics: %s", dir)
		a.pruneWork(5)
		return
	}
	_ = os.RemoveAll(dir)
}

// pruneWork keeps only the newest n work directories.
func (a *Agent) pruneWork(n int) {
	root := filepath.Join(a.cfg.DataDir, "work")
	ents, err := os.ReadDir(root)
	if err != nil || len(ents) <= n {
		return
	}
	type d struct {
		name string
		mod  time.Time
	}
	var ds []d
	for _, e := range ents {
		if info, err := e.Info(); err == nil {
			ds = append(ds, d{e.Name(), info.ModTime()})
		}
	}
	sort.Slice(ds, func(i, j int) bool { return ds[i].mod.After(ds[j].mod) })
	for _, x := range ds[n:] {
		_ = os.RemoveAll(filepath.Join(root, x.name))
	}
}

// ---- outbox: results survive crashes and network outages ----

type outboxEntry struct {
	ProjectID string                       `json:"project_id"`
	AttemptID string                       `json:"attempt_id"`
	JobType   string                       `json:"job_type"`
	Created   time.Time                    `json:"created"`
	Complete  *protocol.JobCompleteRequest `json:"complete,omitempty"`
	Fail      *protocol.JobFailRequest     `json:"fail,omitempty"`
}

func (a *Agent) outboxDir(project string) string {
	if !isUUID(project) {
		return filepath.Join(a.cfg.DataDir, "outbox", "invalid")
	}
	return filepath.Join(a.cfg.DataDir, "outbox", project)
}

func (a *Agent) outboxFile(project, attemptID string) (string, error) {
	if !isUUID(project) || !isUUID(attemptID) {
		return "", fmt.Errorf("unsafe outbox id")
	}
	return a.safeJoin("outbox", project, attemptID+".json")
}

func (a *Agent) writeOutbox(e outboxEntry) error {
	p, err := a.outboxFile(e.ProjectID, e.AttemptID)
	if err != nil {
		return err
	}
	return config.Save(p, e)
}

func (a *Agent) removeOutbox(project, attemptID string) {
	if p, err := a.outboxFile(project, attemptID); err == nil {
		_ = os.Remove(p)
	}
}

func (a *Agent) readOutbox(project string) []outboxEntry {
	ents, _ := os.ReadDir(a.outboxDir(project))
	var out []outboxEntry
	for _, d := range ents {
		if !strings.HasSuffix(d.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(a.outboxDir(project), d.Name()))
		if err != nil {
			continue
		}
		var e outboxEntry
		if json.Unmarshal(b, &e) == nil && isUUID(e.AttemptID) {
			out = append(out, e)
		}
	}
	return out
}

func (a *Agent) dropOutbox(attemptID string) {
	a.mu.Lock()
	id := a.id
	a.mu.Unlock()
	if id == nil {
		return
	}
	a.removeOutbox(id.ProjectID, attemptID)
}

// outboxAttempts reports pending results so their leases stay alive.
func (a *Agent) outboxAttempts() []protocol.AttemptStatus {
	a.mu.Lock()
	id := a.id
	a.mu.Unlock()
	if id == nil {
		return nil
	}
	var out []protocol.AttemptStatus
	for _, e := range a.readOutbox(id.ProjectID) {
		f := int64(0)
		if e.Complete != nil {
			f = e.Complete.FencingToken
		} else if e.Fail != nil {
			f = e.Fail.FencingToken
		}
		out = append(out, protocol.AttemptStatus{AttemptID: e.AttemptID, FencingToken: f, Progress: 1, Stage: "submitting"})
		if len(out) >= 8 {
			break
		}
	}
	return out
}

// submitOutbox sends one entry; it is deleted on success or definitive rejection.
func (a *Agent) submitOutbox(ctx context.Context, cl *client.Client, e outboxEntry) {
	sctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	var err error
	if e.Complete != nil {
		var resp *protocol.JobCompleteResponse
		resp, err = cl.Complete(sctx, e.AttemptID, *e.Complete)
		if err == nil {
			a.logf("info", "jobs", "result for %s: %s%s", e.AttemptID, resp.Status, suffix(resp.Reason))
			a.mu.Lock()
			if resp.Status == protocol.ResultAccepted || resp.Status == protocol.ResultValidating {
				if !resp.Duplicate {
					a.completed++
				}
			} else {
				a.failed++
			}
			a.mu.Unlock()
		}
	} else if e.Fail != nil {
		err = cl.Fail(sctx, e.AttemptID, *e.Fail)
	}
	switch {
	case err == nil:
		a.removeOutbox(e.ProjectID, e.AttemptID)
	case protocol.IsCode(err, protocol.ErrStaleLease), protocol.IsCode(err, protocol.ErrNotFound),
		protocol.IsCode(err, protocol.ErrForbidden), protocol.IsCode(err, protocol.ErrValidation):
		a.logf("warn", "jobs", "controller rejected result for %s: %v (dropping)", e.AttemptID, err)
		a.removeOutbox(e.ProjectID, e.AttemptID)
	default:
		a.logf("warn", "jobs", "could not submit result for %s yet: %v (kept in outbox)", e.AttemptID, err)
	}
}

func suffix(s string) string {
	if s == "" {
		return ""
	}
	return " (" + s + ")"
}

// flushOutbox submits pending entries for the current project within d.
func (a *Agent) flushOutbox(ctx context.Context, d time.Duration) {
	a.mu.Lock()
	cl, id := a.cl, a.id
	a.mu.Unlock()
	if cl == nil || id == nil {
		return
	}
	ctx, cancel := context.WithTimeout(ctx, d)
	defer cancel()
	for _, e := range a.readOutbox(id.ProjectID) {
		if ctx.Err() != nil {
			return
		}
		if t := a.currentTask(); t != nil && t.a.AttemptID == e.AttemptID {
			continue
		}
		a.submitOutbox(ctx, cl, e)
	}
}
