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
	a.mu.Lock()
	loopCtx := context.Background()
	if a.loopCancel != nil {
		loopCtx = a.loopCtxLocked()
	}
	ctx, cancel := context.WithCancel(loopCtx)
	t := &activeTask{a: *job, cancel: cancel, done: make(chan struct{}), phase: "starting", started: time.Now()}
	a.task = t
	a.mu.Unlock()
	a.logf("info", "jobs", "leased %s attempt %s (job %s, attempt #%d)", job.JobType, job.AttemptID, job.JobID, job.AttemptNumber)
	go a.runTask(ctx, cl, t)
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
		id, err = cl.Upload(ctx, art.Path, art.Kind, art.ContentType, func(sent, total int64) {
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
	t.cancel()
	<-pubDone

	entry := outboxEntry{ProjectID: cl.Identity().ProjectID, AttemptID: job.AttemptID, JobType: job.JobType, Created: time.Now()}
	switch {
	case reason == stopDiscard:
		a.logf("info", "jobs", "attempt %s discarded", job.AttemptID)
		a.cleanupWork(job.AttemptID, false)
		return
	case errors.Is(err, executor.ErrStopped) || (err != nil && ctx.Err() != nil):
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
	cacheDir := filepath.Join(a.cfg.DataDir, "cache", cl.Identity().ProjectID)
	if err := os.MkdirAll(cacheDir, 0o700); err != nil {
		return nil, err
	}
	inputs := map[string]string{}
	for i, in := range job.Inputs {
		t.set("downloading", fmt.Sprintf("downloading input %d/%d", i+1, len(job.Inputs)), 0)
		dst := filepath.Join(cacheDir, in.SHA256)
		if err := cl.Download(ctx, in, dst); err != nil {
			if ctx.Err() != nil {
				return nil, executor.ErrStopped
			}
			return nil, &executor.Failure{Code: "DOWNLOAD_FAILED", Message: err.Error(), Retryable: true}
		}
		now := time.Now()
		_ = os.Chtimes(dst, now, now) // LRU timestamp
		key := in.Role
		if key == "" {
			key = in.ID
		}
		inputs[key] = dst
	}
	a.evictCache(cacheDir, s.DiskCacheMB<<20)
	work := filepath.Join(a.cfg.DataDir, "work", job.AttemptID)
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
	dir := filepath.Join(a.cfg.DataDir, "work", attemptID)
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

// evictCache removes least-recently-used files until the cache fits budget.
func (a *Agent) evictCache(dir string, budget int64) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	type f struct {
		path string
		size int64
		mod  time.Time
	}
	var fs []f
	var total int64
	for _, e := range ents {
		if e.IsDir() || strings.HasSuffix(e.Name(), ".part") {
			continue
		}
		if info, err := e.Info(); err == nil {
			fs = append(fs, f{filepath.Join(dir, e.Name()), info.Size(), info.ModTime()})
			total += info.Size()
		}
	}
	if total <= budget {
		return
	}
	sort.Slice(fs, func(i, j int) bool { return fs[i].mod.Before(fs[j].mod) })
	for _, x := range fs[:len(fs)-1] { // never evict the newest (just downloaded)
		if total <= budget {
			break
		}
		if os.Remove(x.path) == nil {
			total -= x.size
			a.logf("info", "cache", "evicted %s from cache", filepath.Base(x.path))
		}
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
	return filepath.Join(a.cfg.DataDir, "outbox", project)
}

func (a *Agent) writeOutbox(e outboxEntry) error {
	return config.Save(filepath.Join(a.outboxDir(e.ProjectID), e.AttemptID+".json"), e)
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
		if json.Unmarshal(b, &e) == nil && e.AttemptID != "" {
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
	_ = os.Remove(filepath.Join(a.outboxDir(id.ProjectID), attemptID+".json"))
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
		_ = os.Remove(filepath.Join(a.outboxDir(e.ProjectID), e.AttemptID+".json"))
	case protocol.IsCode(err, protocol.ErrStaleLease), protocol.IsCode(err, protocol.ErrNotFound),
		protocol.IsCode(err, protocol.ErrForbidden), protocol.IsCode(err, protocol.ErrValidation):
		a.logf("warn", "jobs", "controller rejected result for %s: %v (dropping)", e.AttemptID, err)
		_ = os.Remove(filepath.Join(a.outboxDir(e.ProjectID), e.AttemptID+".json"))
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
