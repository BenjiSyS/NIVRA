package scheduler_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"nivra/internal/controller/scheduler"
	"nivra/internal/shared/builtin"
	"nivra/internal/shared/protocol"
)

func (e *env) user(t *testing.T, name string) string {
	t.Helper()
	var id string
	if err := e.d.DB.QueryRow(context.Background(), `INSERT INTO users (username, password_hash, role) VALUES ($1,'x','member') RETURNING id::text`, name).Scan(&id); err != nil {
		t.Fatal(err)
	}
	return id
}

func (e *env) sleepJob(t *testing.T, by string, prio int) string {
	t.Helper()
	p, _ := json.Marshal(builtin.SleepParams{Seconds: 1})
	nj := scheduler.NewJob{ProjectID: e.project, JobType: builtin.TypeSleep, Params: p, Priority: prio}
	if by != "" {
		nj.CreatedBy = &by
	}
	id, err := e.s.Create(context.Background(), nil, nj)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func createdBy(t *testing.T, e *env, job string) string {
	t.Helper()
	var by *string
	if err := e.d.DB.QueryRow(context.Background(), `SELECT created_by::text FROM jobs WHERE id=$1`, job).Scan(&by); err != nil {
		t.Fatal(err)
	}
	if by == nil {
		return ""
	}
	return *by
}

// Two users: one floods the queue, the other submits a single job later.
// Every free worker goes to whoever has the fewest running attempts, so the
// second user is served right after the first worker is taken.
func TestFairShareBetweenUsers(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	flood, quick := e.user(t, "flooder"), e.user(t, "quick")
	for i := 0; i < 12; i++ {
		e.sleepJob(t, flood, 20)
	}
	late := e.sleepJob(t, quick, 20)
	w1, w2, w3 := e.enroll(t, "w1"), e.enroll(t, "w2"), e.enroll(t, "w3")
	a1, err := e.s.Lease(ctx, e.project, w1)
	if err != nil || a1 == nil {
		t.Fatalf("lease 1: %+v %v", a1, err)
	}
	if got := createdBy(t, e, a1.JobID); got != flood {
		t.Fatalf("first lease went to %s, want the earliest job (flooder)", got)
	}
	a2, err := e.s.Lease(ctx, e.project, w2)
	if err != nil || a2 == nil {
		t.Fatalf("lease 2: %+v %v", a2, err)
	}
	if a2.JobID != late {
		t.Fatalf("second worker got %s (by %s); the quick user's job %s should come first", a2.JobID, createdBy(t, e, a2.JobID), late)
	}
	a3, err := e.s.Lease(ctx, e.project, w3)
	if err != nil || a3 == nil {
		t.Fatalf("lease 3: %+v %v", a3, err)
	}
	if got := createdBy(t, e, a3.JobID); got != flood {
		t.Fatalf("third lease went to %s, want the flooder", got)
	}
}

// Stage priority only orders a user's own work: a high-priority job of a user
// who already has running work does not beat another user's job in the same band.
func TestPriorityBandKeepsStagesInsideTheBand(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a, b := e.user(t, "user-a"), e.user(t, "user-b")
	low := e.sleepJob(t, a, 20)
	e.sleepJob(t, a, 26) // later stage of the same user, same band
	other := e.sleepJob(t, b, 20)
	w1, w2 := e.enroll(t, "w1"), e.enroll(t, "w2")
	x, _ := e.s.Lease(ctx, e.project, w1)
	if x == nil || createdBy(t, e, x.JobID) != a || x.JobID == low {
		t.Fatalf("first lease should be a's higher-priority job inside the band: %+v", x)
	}
	y, _ := e.s.Lease(ctx, e.project, w2)
	if y == nil || y.JobID != other {
		t.Fatalf("second lease must serve user b, got %+v", y)
	}
	// A job in a higher band (priority 30+) still wins over the video band.
	hi := e.sleepJob(t, a, 31)
	w3 := e.enroll(t, "w3")
	z, _ := e.s.Lease(ctx, e.project, w3)
	if z == nil || z.JobID != hi {
		t.Fatalf("a higher band must come first: %+v", z)
	}
}

// Training keeps a reserved share under a flood of videos.
func TestTrainingShareUnderFlood(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	reg := e.s.Registry()
	reg.Register(&scheduler.JobType{Name: "x.train", Category: "training", ValidateParams: func(json.RawMessage) error { return nil }, Validate: acceptAll})
	flood := e.user(t, "flooder")
	for i := 0; i < 12; i++ {
		e.sleepJob(t, flood, 20)
	}
	devs := make([]string, 6)
	for i := range devs {
		devs[i] = e.enroll(t, "w"+string(rune('a'+i)))
		if _, err := e.d.DB.Exec(ctx, `UPDATE worker_capabilities SET executors = executors || '{x.train}' WHERE device_id=$1`, devs[i]); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 3; i++ { // three video leases
		if a, err := e.s.Lease(ctx, e.project, devs[i]); err != nil || a == nil {
			t.Fatalf("video lease %d: %+v %v", i, a, err)
		}
	}
	train, err := e.s.Create(ctx, nil, scheduler.NewJob{ProjectID: e.project, JobType: "x.train", Priority: 10})
	if err != nil {
		t.Fatal(err)
	}
	a, err := e.s.Lease(ctx, e.project, devs[3])
	if err != nil || a == nil || a.JobID != train {
		t.Fatalf("the 4th lease must be the reserved training share, got %+v %v", a, err)
	}
	train2, _ := e.s.Create(ctx, nil, scheduler.NewJob{ProjectID: e.project, JobType: "x.train", Priority: 10})
	b, err := e.s.Lease(ctx, e.project, devs[4])
	if err != nil || b == nil || b.JobID == train2 {
		t.Fatalf("right after a training lease the next one is a video again, got %+v %v", b, err)
	}
}

// Refunded attempts (lost worker / expired lease) are capped per job.
func TestLeaseExpiryRefundIsCapped(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	p, _ := json.Marshal(builtin.SleepParams{Seconds: 1})
	id, err := e.s.Create(ctx, nil, scheduler.NewJob{ProjectID: e.project, JobType: builtin.TypeSleep, Params: p, MaxAttempts: 1})
	if err != nil {
		t.Fatal(err)
	}
	dev := e.enroll(t, "w1")
	expire := func() {
		if _, err := e.d.DB.Exec(ctx, `UPDATE job_attempts SET lease_expires_at = now() - interval '1 minute' WHERE job_id=$1 AND state IN ('leased','running')`, id); err != nil {
			t.Fatal(err)
		}
		if _, err := e.s.ReapOnce(ctx); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 5; i++ {
		a, err := e.s.Lease(ctx, e.project, dev)
		if err != nil || a == nil {
			t.Fatalf("lease %d after refund: %+v %v", i, a, err)
		}
		expire()
		v, _ := e.s.GetJob(ctx, e.project, id)
		if v.Status != scheduler.JobReady {
			t.Fatalf("expiry %d: job is %s, want ready (attempt refunded)", i+1, v.Status)
		}
	}
	if a, _ := e.s.Lease(ctx, e.project, dev); a == nil {
		t.Fatal("sixth lease should still happen")
	}
	expire()
	v, _ := e.s.GetJob(ctx, e.project, id)
	if v.Status != scheduler.JobFailed {
		t.Fatalf("after the refund cap the job must fail, got %s", v.Status)
	}
}

// A retried video stage backs off before it is offered again, and the job view
// explains why a job nobody can run is waiting.
func TestRetryBackoffAndWaitingReason(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	e.s.Registry().Register(&scheduler.JobType{Name: "ai.refine_segment", Category: "generation", ValidateParams: func(json.RawMessage) error { return nil }, Validate: acceptAll})
	dev := e.enroll(t, "w1")
	if _, err := e.d.DB.Exec(ctx, `UPDATE worker_capabilities SET executors = executors || '{ai.refine_segment}', vram_budget_mb=2048 WHERE device_id=$1`, dev); err != nil {
		t.Fatal(err)
	}
	if _, err := e.d.DB.Exec(ctx, `UPDATE devices SET last_seen_at = now() WHERE id=$1`, dev); err != nil {
		t.Fatal(err)
	}
	id, err := e.s.Create(ctx, nil, scheduler.NewJob{ProjectID: e.project, JobType: "ai.refine_segment", MaxAttempts: 6, ReqVRAMMB: 8000})
	if err != nil {
		t.Fatal(err)
	}
	v, _ := e.s.GetJob(ctx, e.project, id)
	if !strings.Contains(v.WaitingReason, "graphics memory") {
		t.Fatalf("waiting reason = %q", v.WaitingReason)
	}
	// A job the device can run, failing once.
	id2, _ := e.s.Create(ctx, nil, scheduler.NewJob{ProjectID: e.project, JobType: "ai.refine_segment", MaxAttempts: 6, ReqVRAMMB: 1000})
	a, err := e.s.Lease(ctx, e.project, dev)
	if err != nil || a == nil || a.JobID != id2 {
		t.Fatalf("lease: %+v %v", a, err)
	}
	if err := e.s.Fail(ctx, dev, a.AttemptID, protocolFail(a.FencingToken)); err != nil {
		t.Fatal(err)
	}
	if again, _ := e.s.Lease(ctx, e.project, dev); again != nil {
		t.Fatalf("the retried stage must wait out its back-off, got %+v", again)
	}
	v2, _ := e.s.GetJob(ctx, e.project, id2)
	if v2.Status != scheduler.JobReady || !strings.Contains(v2.WaitingReason, "tried again") {
		t.Fatalf("view after failure: %+v", v2)
	}
	if _, err := e.d.DB.Exec(ctx, `UPDATE jobs SET available_at = now() - interval '1 second' WHERE id=$1`, id2); err != nil {
		t.Fatal(err)
	}
	if again, _ := e.s.Lease(ctx, e.project, dev); again == nil || again.JobID != id2 {
		t.Fatalf("after the back-off the job is offered again, got %+v", again)
	}
}

func protocolFail(fencing int64) protocol.JobFailRequest {
	return protocol.JobFailRequest{FencingToken: fencing, Code: "INTERNAL", Message: "boom", Retryable: true}
}

func acceptAll(context.Context, scheduler.ValidationInput) (scheduler.Verdict, error) {
	return scheduler.Verdict{Accept: true}, nil
}
