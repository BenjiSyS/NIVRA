package scheduler_test

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"nivra/internal/controller/core"
	"nivra/internal/controller/enrollment"
	"nivra/internal/controller/projects"
	"nivra/internal/controller/scheduler"
	"nivra/internal/shared/builtin"
	"nivra/internal/shared/protocol"
	"nivra/internal/testutil"
)

type env struct {
	d       *core.Deps
	s       *scheduler.Scheduler
	project string
	code    string
}

func testutilDeps(t *testing.T) *core.Deps { return testutil.Deps(t) }

func projectsCreate(t *testing.T, d *core.Deps) (string, string, error) {
	p, ci, err := projects.New(d).Create(context.Background(), core.System, "T2")
	if err != nil {
		t.Fatal(err)
	}
	return p.ID, ci.Code, nil
}

func setup(t *testing.T) *env {
	t.Helper()
	d := testutil.Deps(t)
	reg := scheduler.NewRegistry()
	scheduler.RegisterBuiltins(reg)
	p, ci, err := projects.New(d).Create(context.Background(), core.System, "Test")
	if err != nil {
		t.Fatal(err)
	}
	return &env{d: d, s: scheduler.New(d, reg), project: p.ID, code: ci.Code}
}

// enroll performs the real enrollment protocol and reports capabilities.
func (e *env) enroll(t *testing.T, name string) string {
	t.Helper()
	ctx := context.Background()
	pub, priv, _ := ed25519.GenerateKey(nil)
	en := enrollment.New(e.d)
	b, err := en.Begin(ctx, protocol.EnrollBeginRequest{Code: e.code, PublicKey: pub,
		Device: protocol.DeviceInfo{Name: name, OS: "linux", Arch: "amd64"}}, "test")
	if err != nil {
		t.Fatal(err)
	}
	c, err := en.Complete(ctx, protocol.EnrollCompleteRequest{EnrollmentID: b.EnrollmentID,
		Signature: ed25519.Sign(priv, protocol.EnrollProofMessage(b.EnrollmentID, b.Nonce, b.ProjectID))}, "test")
	if err != nil {
		t.Fatal(err)
	}
	_, err = e.d.DB.Exec(ctx, `INSERT INTO worker_capabilities (device_id, executors, ram_budget_mb, ram_mb) VALUES ($1, $2, 4096, 8192)`,
		c.DeviceID, []string{builtin.TypeSHA256Chain, builtin.TypeSleep})
	if err != nil {
		t.Fatal(err)
	}
	return c.DeviceID
}

func (e *env) chainJob(t *testing.T, iters int64) (string, string) {
	t.Helper()
	seed := hex.EncodeToString(make([]byte, 32))
	params, _ := json.Marshal(builtin.SHA256ChainParams{Seed: seed, Iterations: iters})
	id, err := e.s.Create(context.Background(), nil, scheduler.NewJob{ProjectID: e.project, JobType: builtin.TypeSHA256Chain, Params: params})
	if err != nil {
		t.Fatal(err)
	}
	digest, _ := builtin.SHA256Chain(context.Background(), builtin.SHA256ChainParams{Seed: seed, Iterations: iters}, nil)
	return id, digest
}

func result(digest string) json.RawMessage {
	b, _ := json.Marshal(builtin.SHA256ChainResult{Digest: digest})
	return b
}

func contributions(t *testing.T, e *env, device string) (n int, credit float64) {
	t.Helper()
	if err := e.d.DB.QueryRow(context.Background(), `SELECT count(*), COALESCE(sum(credit),0) FROM contributions WHERE device_id=$1`, device).
		Scan(&n, &credit); err != nil {
		t.Fatal(err)
	}
	return
}

func TestHappyPathAndDuplicateSubmission(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	dev := e.enroll(t, "w1")
	jobID, digest := e.chainJob(t, 1000)
	a, err := e.s.Lease(ctx, e.project, dev)
	if err != nil || a == nil || a.JobID != jobID {
		t.Fatalf("lease: %+v %v", a, err)
	}
	if again, _ := e.s.Lease(ctx, e.project, dev); again != nil {
		t.Fatal("device leased a second job while one is active")
	}
	if err := e.s.Ack(ctx, dev, a.AttemptID, a.FencingToken); err != nil {
		t.Fatal(err)
	}
	if err := e.s.Progress(ctx, dev, a.AttemptID, protocol.JobProgressRequest{FencingToken: a.FencingToken, Progress: 0.5, Stage: "hash"}); err != nil {
		t.Fatal(err)
	}
	req := protocol.JobCompleteRequest{FencingToken: a.FencingToken, Result: result(digest)}
	r1, err := e.s.Complete(ctx, dev, a.AttemptID, req)
	if err != nil || r1.Status != protocol.ResultAccepted || r1.Duplicate {
		t.Fatalf("complete: %+v %v", r1, err)
	}
	r2, err := e.s.Complete(ctx, dev, a.AttemptID, req)
	if err != nil || !r2.Duplicate || r2.Status != protocol.ResultAccepted {
		t.Fatalf("duplicate: %+v %v", r2, err)
	}
	if n, c := contributions(t, e, dev); n != 1 || c != 0.001 {
		t.Fatalf("contributions = %d / %v, want exactly 1 / 0.001", n, c)
	}
	j, _ := e.s.GetJob(ctx, e.project, jobID)
	if j.Status != scheduler.JobCompleted {
		t.Fatalf("job status %s", j.Status)
	}
}

func TestWrongResultRejectedAndRetried(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	dev := e.enroll(t, "w1")
	jobID, _ := e.chainJob(t, 500)
	a, _ := e.s.Lease(ctx, e.project, dev)
	r, err := e.s.Complete(ctx, dev, a.AttemptID, protocol.JobCompleteRequest{FencingToken: a.FencingToken, Result: result("00")})
	if err != nil || r.Status != protocol.ResultRejected {
		t.Fatalf("want rejected, got %+v %v", r, err)
	}
	j, _ := e.s.GetJob(ctx, e.project, jobID)
	if j.Status != scheduler.JobReady {
		t.Fatalf("rejected job should be ready for retry, got %s", j.Status)
	}
	// The rejecting device is not given the same job again.
	if again, _ := e.s.Lease(ctx, e.project, dev); again != nil {
		t.Fatal("job re-leased to the device whose result was rejected")
	}
	if n, _ := contributions(t, e, dev); n != 0 {
		t.Fatal("rejected work was credited")
	}
}

func TestExpiredLeaseIsFencedAndReassigned(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	slow, fast := e.enroll(t, "slow"), e.enroll(t, "fast")
	_, digest := e.chainJob(t, 200)
	a1, _ := e.s.Lease(ctx, e.project, slow)
	if a1 == nil {
		t.Fatal("no lease")
	}
	if _, err := e.d.DB.Exec(ctx, `UPDATE job_attempts SET lease_expires_at = now() - interval '1 second' WHERE id=$1`, a1.AttemptID); err != nil {
		t.Fatal(err)
	}
	if n, err := e.s.ReapOnce(ctx); err != nil || n != 1 {
		t.Fatalf("reap: %d %v", n, err)
	}
	a2, _ := e.s.Lease(ctx, e.project, fast)
	if a2 == nil || a2.FencingToken <= a1.FencingToken {
		t.Fatalf("reassignment failed or fencing token not increasing: %+v", a2)
	}
	_, err := e.s.Complete(ctx, slow, a1.AttemptID, protocol.JobCompleteRequest{FencingToken: a1.FencingToken, Result: result(digest)})
	if !errors.Is(err, scheduler.ErrStale) {
		t.Fatalf("late result from expired lease: want ErrStale, got %v", err)
	}
	r, err := e.s.Complete(ctx, fast, a2.AttemptID, protocol.JobCompleteRequest{FencingToken: a2.FencingToken, Result: result(digest)})
	if err != nil || r.Status != protocol.ResultAccepted {
		t.Fatalf("%+v %v", r, err)
	}
	if n, _ := contributions(t, e, slow); n != 0 {
		t.Fatal("stale worker credited")
	}
	// Forged fencing token on someone else's attempt.
	_, err = e.s.Complete(ctx, slow, a2.AttemptID, protocol.JobCompleteRequest{FencingToken: a2.FencingToken, Result: result(digest)})
	if !errors.Is(err, scheduler.ErrNotYours) {
		t.Fatalf("want ErrNotYours, got %v", err)
	}
}

func TestReconcileReleasesLostAttempt(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	dev := e.enroll(t, "w1")
	jobID, _ := e.chainJob(t, 100)
	a, _ := e.s.Lease(ctx, e.project, dev)
	// Within the grace period an unreported attempt is kept.
	if _, err := e.s.Reconcile(ctx, e.project, dev, nil); err != nil {
		t.Fatal(err)
	}
	if j, _ := e.s.GetJob(ctx, e.project, jobID); j.Status != scheduler.JobLeased {
		t.Fatalf("status %s", j.Status)
	}
	e.d.DB.Exec(ctx, `UPDATE job_attempts SET leased_at = now() - interval '1 minute' WHERE id=$1`, a.AttemptID)
	if _, err := e.s.Reconcile(ctx, e.project, dev, nil); err != nil {
		t.Fatal(err)
	}
	if j, _ := e.s.GetJob(ctx, e.project, jobID); j.Status != scheduler.JobReady {
		t.Fatalf("lost attempt not released: %s", j.Status)
	}
	invalid, _ := e.s.Reconcile(ctx, e.project, dev, []protocol.AttemptStatus{{AttemptID: a.AttemptID, FencingToken: a.FencingToken}})
	if len(invalid) != 1 {
		t.Fatalf("worker should be told its stale attempt is invalid, got %v", invalid)
	}
}

func TestConcurrentLeasingNeverDoubleAssigns(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	const workers, jobs = 12, 30
	var devs []string
	for i := 0; i < workers; i++ {
		devs = append(devs, e.enroll(t, fmt.Sprintf("w%d", i)))
	}
	for i := 0; i < jobs; i++ {
		e.chainJob(t, 10)
	}
	var mu sync.Mutex
	seen := map[string]string{}
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for _, d := range devs {
		wg.Add(1)
		go func(dev string) {
			defer wg.Done()
			for {
				a, err := e.s.Lease(ctx, e.project, dev)
				if err != nil {
					errs <- err
					return
				}
				if a == nil {
					return
				}
				mu.Lock()
				if prev, dup := seen[a.JobID]; dup {
					mu.Unlock()
					errs <- fmt.Errorf("job %s leased to %s and %s", a.JobID, prev, dev)
					return
				}
				seen[a.JobID] = dev
				mu.Unlock()
				var p builtin.SHA256ChainParams
				json.Unmarshal(a.Params, &p)
				dg, _ := builtin.SHA256Chain(ctx, p, nil)
				if _, err := e.s.Complete(ctx, dev, a.AttemptID, protocol.JobCompleteRequest{FencingToken: a.FencingToken, Result: result(dg)}); err != nil {
					errs <- err
					return
				}
			}
		}(d)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	if len(seen) != jobs {
		t.Fatalf("completed %d jobs, want %d", len(seen), jobs)
	}
	var accepted int
	e.d.DB.QueryRow(ctx, `SELECT count(*) FROM contributions`).Scan(&accepted)
	if accepted != jobs {
		t.Fatalf("contributions %d, want %d", accepted, jobs)
	}
}

func TestStopRequeuesAndCancel(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	dev := e.enroll(t, "w1")
	params, _ := json.Marshal(builtin.SleepParams{Seconds: 1})
	jobID, err := e.s.Create(ctx, nil, scheduler.NewJob{ProjectID: e.project, JobType: builtin.TypeSleep, Params: params, MaxAttempts: 1})
	if err != nil {
		t.Fatal(err)
	}
	a, _ := e.s.Lease(ctx, e.project, dev)
	if err := e.s.Fail(ctx, dev, a.AttemptID, protocol.JobFailRequest{FencingToken: a.FencingToken, Code: "STOPPED"}); err != nil {
		t.Fatal(err)
	}
	if j, _ := e.s.GetJob(ctx, e.project, jobID); j.Status != scheduler.JobReady || j.AttemptCount != 0 {
		t.Fatalf("stopped job should be requeued without burning an attempt: %s %d", j.Status, j.AttemptCount)
	}
	a2, err := e.s.Lease(ctx, e.project, dev)
	if err != nil || a2 == nil {
		t.Fatalf("re-lease after stop: %v", err)
	}
	if err := e.s.Cancel(ctx, core.System, e.project, jobID); err != nil {
		t.Fatal(err)
	}
	_, err = e.s.Complete(ctx, dev, a2.AttemptID, protocol.JobCompleteRequest{FencingToken: a2.FencingToken, Result: json.RawMessage(`{"slept_ms":1000}`)})
	if !errors.Is(err, scheduler.ErrStale) {
		t.Fatalf("result after cancel: want ErrStale, got %v", err)
	}
	time.Sleep(10 * time.Millisecond)
}

func TestTrustedJobsOnlyGoToTrustedDevices(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	dev := e.enroll(t, "untrusted")
	params, _ := json.Marshal(builtin.SleepParams{Seconds: 1})
	if _, err := e.s.Create(ctx, nil, scheduler.NewJob{ProjectID: e.project, JobType: builtin.TypeSleep, Params: params, ReqTrusted: true}); err != nil {
		t.Fatal(err)
	}
	if a, _ := e.s.Lease(ctx, e.project, dev); a != nil {
		t.Fatal("untrusted device leased a trusted-only job")
	}
	e.d.DB.Exec(ctx, `UPDATE devices SET trusted=true WHERE id=$1`, dev)
	if a, err := e.s.Lease(ctx, e.project, dev); err != nil || a == nil {
		t.Fatalf("trusted device could not lease: %v", err)
	}
}

func TestValidatorFailuresAreBounded(t *testing.T) {
	d := testutilDeps(t)
	reg := scheduler.NewRegistry()
	calls := 0
	reg.Register(&scheduler.JobType{Name: "test.flaky", Category: "test", Async: true,
		ValidateParams: func(json.RawMessage) error { return nil },
		Validate: func(context.Context, scheduler.ValidationInput) (scheduler.Verdict, error) {
			calls++
			return scheduler.Verdict{}, errors.New("validator infrastructure down")
		}})
	e := &env{d: d, s: scheduler.New(d, reg)}
	p, ci, _ := projectsCreate(t, d)
	e.project, e.code = p, ci
	dev := e.enroll(t, "w")
	e.d.DB.Exec(context.Background(), `UPDATE worker_capabilities SET executors = executors || '{test.flaky}' WHERE device_id=$1`, dev)
	ctx := context.Background()
	jobID, err := e.s.Create(ctx, nil, scheduler.NewJob{ProjectID: e.project, JobType: "test.flaky", MaxAttempts: 1})
	if err != nil {
		t.Fatal(err)
	}
	a, _ := e.s.Lease(ctx, e.project, dev)
	if r, err := e.s.Complete(ctx, dev, a.AttemptID, protocol.JobCompleteRequest{FencingToken: a.FencingToken}); err != nil || r.Status != protocol.ResultValidating {
		t.Fatalf("%+v %v", r, err)
	}
	for i := 0; i < 5; i++ {
		e.s.ValidatePendingOnce(ctx)
	}
	j, _ := e.s.GetJob(ctx, e.project, jobID)
	if j.Status != scheduler.JobFailed || calls != 3 {
		t.Fatalf("want job failed after 3 validator errors, got %s after %d calls", j.Status, calls)
	}
}
