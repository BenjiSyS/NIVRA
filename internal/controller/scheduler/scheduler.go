// Package scheduler is the durable job scheduler: jobs and leased attempts
// live in PostgreSQL, every attempt carries a monotonically increasing
// fencing token, and only the current attempt of a job can commit a result.
package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/controller/events"
	"nivra/internal/shared/protocol"
)

// Errors returned to the API layer.
var (
	ErrUnknownJobType = errors.New("unknown job type")
	ErrStale          = errors.New("attempt is no longer current (lease expired, reassigned or cancelled)")
	ErrNotYours       = errors.New("attempt belongs to another device")
	ErrNotFound       = errors.New("job or attempt not found")
	ErrBadState       = errors.New("operation not allowed in the current state")
	ErrBadArtifact    = errors.New("result artifact missing, incomplete or not owned by this device")
)

// Scheduler owns job state.
type Scheduler struct {
	d     *core.Deps
	reg   *Registry
	valCh chan struct{}
}

// New creates a scheduler.
func New(d *core.Deps, reg *Registry) *Scheduler {
	return &Scheduler{d: d, reg: reg, valCh: make(chan struct{}, 1)}
}

// Registry exposes the job type registry.
func (s *Scheduler) Registry() *Registry { return s.reg }

// NewJob describes a job to enqueue.
type NewJob struct {
	ProjectID string
	JobType   string
	Params    json.RawMessage
	Inputs    []protocol.ArtifactRef // only ID and Role are used
	ReqRAMMB  int
	ReqVRAMMB int
	ReqGPU    bool
	ReqOS     string
	// ReqTrusted restricts the job to devices an administrator marked trusted.
	ReqTrusted     bool
	Priority       int
	MaxAttempts    int
	LeaseSeconds   int
	Deadline       *time.Time
	IdempotencyKey string
	ModelVersion   *string
	TrainingRunID  *string
	CreatedBy      *string
}

// Create validates and enqueues a job (status ready). With an idempotency
// key, re-submitting returns the existing job instead of a duplicate.
func (s *Scheduler) Create(ctx context.Context, q core.Querier, j NewJob) (string, error) {
	t, ok := s.reg.Get(j.JobType)
	if !ok {
		return "", fmt.Errorf("%w: %s", ErrUnknownJobType, j.JobType)
	}
	if len(j.Params) == 0 {
		j.Params = json.RawMessage(`{}`)
	}
	if len(j.Params) > 64<<10 {
		return "", errors.New("params too large")
	}
	if err := t.ValidateParams(j.Params); err != nil {
		return "", err
	}
	if j.MaxAttempts == 0 {
		j.MaxAttempts = 3
	}
	if j.LeaseSeconds == 0 {
		j.LeaseSeconds = 90
	}
	if q == nil {
		q = s.d.DB
	}
	ids := make([]string, 0, len(j.Inputs))
	roles := make([]string, 0, len(j.Inputs))
	for _, in := range j.Inputs {
		ids = append(ids, in.ID)
		roles = append(roles, in.Role)
	}
	var reqOS, idem any
	if j.ReqOS != "" {
		reqOS = j.ReqOS
	}
	if j.IdempotencyKey != "" {
		idem = j.IdempotencyKey
	}
	var id string
	err := q.QueryRow(ctx, `INSERT INTO jobs (project_id, job_type, category, model_version, training_run_id, params,
			input_artifacts, input_roles, req_ram_mb, req_vram_mb, req_gpu, req_os, priority, status, max_attempts,
			lease_seconds, deadline_at, idempotency_key, created_by, req_trusted)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,'ready',$14,$15,$16,$17,$18,$19)
		ON CONFLICT (project_id, idempotency_key) DO UPDATE SET updated_at = jobs.updated_at
		RETURNING id`,
		j.ProjectID, j.JobType, t.Category, j.ModelVersion, j.TrainingRunID, j.Params, ids, roles,
		j.ReqRAMMB, j.ReqVRAMMB, j.ReqGPU, reqOS, j.Priority, j.MaxAttempts, j.LeaseSeconds, j.Deadline, idem, j.CreatedBy, j.ReqTrusted).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("create job: %w", err)
	}
	s.d.Events.NotifyJobs(j.ProjectID)
	s.publishJob(ctx, j.ProjectID, id)
	return id, nil
}

// Lease assigns the best matching ready job to a device, or returns nil.
func (s *Scheduler) Lease(ctx context.Context, projectID, deviceID string) (*protocol.JobAssignment, error) {
	var a *protocol.JobAssignment
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var paused bool
		var status, devOS string
		var trusted bool
		// Locking the device row serialises concurrent lease calls per device.
		if err := tx.QueryRow(ctx, `SELECT paused, status, os, trusted FROM devices WHERE id=$1 AND project_id=$2 FOR UPDATE`,
			deviceID, projectID).Scan(&paused, &status, &devOS, &trusted); err != nil {
			return err
		}
		if paused || status != "active" {
			return nil
		}
		var active int
		if err := tx.QueryRow(ctx, `SELECT count(*) FROM job_attempts WHERE device_id=$1 AND state = ANY($2)`,
			deviceID, activeAttemptStates).Scan(&active); err != nil {
			return err
		}
		if active > 0 {
			return nil
		}
		var executors []string
		var ram, vram int64
		err := tx.QueryRow(ctx, `SELECT executors, ram_budget_mb, vram_budget_mb FROM worker_capabilities WHERE device_id=$1`,
			deviceID).Scan(&executors, &ram, &vram)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil // capabilities not reported yet
		}
		if err != nil {
			return err
		}
		var jobID string
		err = tx.QueryRow(ctx, `SELECT j.id FROM jobs j
			WHERE j.project_id=$1 AND j.status='ready'
			  AND j.job_type = ANY($2)
			  AND j.req_ram_mb <= $3 AND j.req_vram_mb <= $4 AND (NOT j.req_gpu OR $4 > 0)
			  AND (j.req_os IS NULL OR j.req_os = $5)
			  AND (j.deadline_at IS NULL OR j.deadline_at > now())
			  AND NOT EXISTS (SELECT 1 FROM job_attempts x WHERE x.job_id=j.id AND x.device_id=$6 AND x.state='rejected')
			  AND (NOT j.req_trusted OR $7)
			ORDER BY j.priority DESC, j.created_at
			LIMIT 1 FOR UPDATE OF j SKIP LOCKED`, projectID, executors, ram, vram, devOS, deviceID, trusted).Scan(&jobID)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil
		}
		if err != nil {
			return err
		}
		a = &protocol.JobAssignment{JobID: jobID}
		var inputIDs []string
		var roles []string
		// attempt_count is the retry budget (refunded for operator stops);
		// attempt_number is a strictly increasing sequence per job.
		err = tx.QueryRow(ctx, `UPDATE jobs SET status='leased', attempt_count=attempt_count+1, updated_at=now()
			WHERE id=$1 RETURNING job_type, category, params, input_artifacts::text[], input_roles, COALESCE(model_version::text, ''),
			lease_seconds`, jobID).Scan(&a.JobType, &a.Category, &a.Params, &inputIDs, &roles, &a.ModelVersion, &a.LeaseSeconds)
		if err != nil {
			return err
		}
		if err := tx.QueryRow(ctx, `SELECT COALESCE(max(attempt_number), 0) + 1 FROM job_attempts WHERE job_id=$1`, jobID).
			Scan(&a.AttemptNumber); err != nil {
			return err
		}
		err = tx.QueryRow(ctx, `INSERT INTO job_attempts (job_id, device_id, attempt_number, fencing_token, state, lease_expires_at)
			VALUES ($1, $2, $3, nextval('fencing_seq'), 'leased', now() + make_interval(secs => $4))
			RETURNING id, fencing_token, lease_expires_at`, jobID, deviceID, a.AttemptNumber, a.LeaseSeconds).
			Scan(&a.AttemptID, &a.FencingToken, &a.LeaseExpiresAt)
		if err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE jobs SET current_attempt_id=$2 WHERE id=$1`, jobID, a.AttemptID); err != nil {
			return err
		}
		a.Inputs = make([]protocol.ArtifactRef, 0, len(inputIDs))
		for i, aid := range inputIDs {
			ref := protocol.ArtifactRef{ID: aid}
			if i < len(roles) {
				ref.Role = roles[i]
			}
			if err := tx.QueryRow(ctx, `SELECT kind, sha256, size_bytes FROM artifacts WHERE id=$1 AND state='complete'`, aid).
				Scan(&ref.Kind, &ref.SHA256, &ref.Size); err != nil {
				return fmt.Errorf("job %s input artifact %s unavailable: %w", jobID, aid, err)
			}
			a.Inputs = append(a.Inputs, ref)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if a != nil {
		s.publishJob(ctx, projectID, a.JobID)
	}
	return a, nil
}

// attemptRow is the locked state of an attempt and its job.
type attemptRow struct {
	attemptID, jobID, projectID, deviceID, state, jobStatus, jobType string
	fencing                                                          int64
	current                                                          *string
	leaseSeconds, attemptCount, maxAttempts                          int
}

func lockAttempt(ctx context.Context, tx pgx.Tx, attemptID string) (*attemptRow, error) {
	r := &attemptRow{attemptID: attemptID}
	err := tx.QueryRow(ctx, `SELECT a.job_id, j.project_id, a.device_id, a.state, j.status, j.job_type, a.fencing_token,
			j.current_attempt_id::text, j.lease_seconds, j.attempt_count, j.max_attempts
		FROM job_attempts a JOIN jobs j ON j.id = a.job_id
		WHERE a.id=$1 FOR UPDATE OF a, j`, attemptID).
		Scan(&r.jobID, &r.projectID, &r.deviceID, &r.state, &r.jobStatus, &r.jobType, &r.fencing, &r.current,
			&r.leaseSeconds, &r.attemptCount, &r.maxAttempts)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	return r, err
}

// checkCurrent enforces ownership, fencing and that the attempt still holds the lease.
func (r *attemptRow) checkCurrent(deviceID string, fencing int64) error {
	if r.deviceID != deviceID {
		return ErrNotYours
	}
	if r.fencing != fencing || r.current == nil || *r.current != r.attemptID {
		return ErrStale
	}
	switch r.state {
	case AttLeased, AttRunning, AttUploading:
		return nil
	default:
		return ErrStale
	}
}

// Ack marks a leased attempt as running.
func (s *Scheduler) Ack(ctx context.Context, deviceID, attemptID string, fencing int64) error {
	var projectID, jobID string
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		r, err := lockAttempt(ctx, tx, attemptID)
		if err != nil {
			return err
		}
		if err := r.checkCurrent(deviceID, fencing); err != nil {
			return err
		}
		projectID, jobID = r.projectID, r.jobID
		if r.state != AttLeased {
			return nil // idempotent re-ack
		}
		if _, err := tx.Exec(ctx, `UPDATE job_attempts SET state='running', started_at=now(),
			lease_expires_at=now() + make_interval(secs => $2) WHERE id=$1`, attemptID, r.leaseSeconds); err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `UPDATE jobs SET status='running', updated_at=now() WHERE id=$1`, r.jobID)
		return err
	})
	if err == nil {
		s.publishJob(ctx, projectID, jobID)
	}
	return err
}

// Progress records progress and extends the lease.
func (s *Scheduler) Progress(ctx context.Context, deviceID, attemptID string, p protocol.JobProgressRequest) error {
	if p.Progress < 0 || p.Progress > 1 {
		return fmt.Errorf("progress must be 0..1")
	}
	state := p.State
	if state == "" {
		state = AttRunning
	}
	if state != AttRunning && state != AttUploading {
		return fmt.Errorf("state must be running or uploading")
	}
	var projectID, jobID string
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		r, err := lockAttempt(ctx, tx, attemptID)
		if err != nil {
			return err
		}
		if err := r.checkCurrent(deviceID, p.FencingToken); err != nil {
			return err
		}
		projectID, jobID = r.projectID, r.jobID
		if _, err := tx.Exec(ctx, `UPDATE job_attempts SET state=$2, progress=$3, stage=$4,
			started_at=COALESCE(started_at, now()), lease_expires_at=now() + make_interval(secs => $5) WHERE id=$1`,
			attemptID, state, p.Progress, truncate(p.Stage, 128), r.leaseSeconds); err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `UPDATE jobs SET status=$2, updated_at=now() WHERE id=$1`, r.jobID, state)
		return err
	})
	if err == nil {
		s.publishJob(ctx, projectID, jobID)
	}
	return err
}

// Fail records a failed attempt; the job is retried while attempts remain.
func (s *Scheduler) Fail(ctx context.Context, deviceID, attemptID string, f protocol.JobFailRequest) error {
	var projectID, jobID string
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		r, err := lockAttempt(ctx, tx, attemptID)
		if err != nil {
			return err
		}
		if err := r.checkCurrent(deviceID, f.FencingToken); err != nil {
			if errors.Is(err, ErrStale) && r.state == AttFailed {
				return nil // duplicate failure report
			}
			return err
		}
		projectID, jobID = r.projectID, r.jobID
		msg := truncate(strings.TrimSpace(f.Code+": "+f.Message), 1000)
		if _, err := tx.Exec(ctx, `UPDATE job_attempts SET state='failed', completed_at=now(), error=$2 WHERE id=$1`, attemptID, msg); err != nil {
			return err
		}
		// A stopped task is requeued without counting against the job.
		requeue := (f.Retryable && r.attemptCount < r.maxAttempts) || f.Code == "STOPPED"
		next := JobFailed
		if requeue {
			next = JobReady
		}
		if f.Code == "STOPPED" {
			if _, err := tx.Exec(ctx, `UPDATE jobs SET attempt_count=GREATEST(attempt_count-1, 0) WHERE id=$1`, r.jobID); err != nil {
				return err
			}
		}
		_, err = tx.Exec(ctx, `UPDATE jobs SET status=$2, current_attempt_id=NULL, failure_reason=$3, updated_at=now(),
			completed_at=CASE WHEN $2='failed' THEN now() ELSE NULL END WHERE id=$1`, r.jobID, next, msg)
		return err
	})
	if err == nil {
		s.d.Events.NotifyJobs(projectID)
		s.publishJob(ctx, projectID, jobID)
	}
	return err
}

// Cancel cancels a job and its active attempt (admin action).
func (s *Scheduler) Cancel(ctx context.Context, actor core.Actor, projectID, jobID string) error {
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var status string
		if err := tx.QueryRow(ctx, `SELECT status FROM jobs WHERE id=$1 AND project_id=$2 FOR UPDATE`, jobID, projectID).Scan(&status); err != nil {
			return ErrNotFound
		}
		if IsTerminalJob(status) {
			return ErrBadState
		}
		if _, err := tx.Exec(ctx, `UPDATE job_attempts SET state='cancelled', completed_at=now(), error='cancelled by operator'
			WHERE job_id=$1 AND state = ANY($2)`, jobID, append(activeAttemptStates, AttValidating)); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE job_results SET validation_status='rejected', rejection_reason='job cancelled', validated_at=now()
			WHERE job_id=$1 AND validation_status='pending'`, jobID); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `UPDATE jobs SET status='cancelled', current_attempt_id=NULL, updated_at=now(), completed_at=now() WHERE id=$1`, jobID)
		return err
	})
	if err == nil {
		s.d.Audit(ctx, nil, actor, "job.cancel", "job:"+jobID, "success", nil)
		s.publishJob(ctx, projectID, jobID)
	}
	return err
}

// Retry moves a failed job back to ready with a fresh attempt budget.
func (s *Scheduler) Retry(ctx context.Context, actor core.Actor, projectID, jobID string) error {
	tag, err := s.d.DB.Exec(ctx, `UPDATE jobs SET status='ready', max_attempts=attempt_count + 3, failure_reason='', completed_at=NULL, updated_at=now()
		WHERE id=$1 AND project_id=$2 AND status='failed'`, jobID, projectID)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrBadState
	}
	s.d.Audit(ctx, nil, actor, "job.retry", "job:"+jobID, "success", nil)
	s.d.Events.NotifyJobs(projectID)
	s.publishJob(ctx, projectID, jobID)
	return nil
}

// CheckActive verifies that attemptID is the device's current, unexpired
// attempt with the given fencing token.
func (s *Scheduler) CheckActive(ctx context.Context, deviceID, attemptID string, fencing int64) error {
	var dev, state string
	var tok int64
	var current *string
	err := s.d.DB.QueryRow(ctx, `SELECT a.device_id, a.state, a.fencing_token, j.current_attempt_id::text
		FROM job_attempts a JOIN jobs j ON j.id=a.job_id WHERE a.id=$1`, attemptID).Scan(&dev, &state, &tok, &current)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrNotFound
	}
	if err != nil {
		return err
	}
	r := &attemptRow{attemptID: attemptID, deviceID: dev, state: state, fencing: tok, current: current}
	return r.checkCurrent(deviceID, fencing)
}

// ActiveAttempts returns the device's attempts that still hold a lease.
func (s *Scheduler) ActiveAttempts(ctx context.Context, deviceID string) (map[string]int64, error) {
	rows, err := s.d.DB.Query(ctx, `SELECT id, fencing_token FROM job_attempts WHERE device_id=$1 AND state = ANY($2)`,
		deviceID, activeAttemptStates)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]int64{}
	for rows.Next() {
		var id string
		var f int64
		if err := rows.Scan(&id, &f); err != nil {
			return nil, err
		}
		out[id] = f
	}
	return out, rows.Err()
}

// Reconcile is called on every heartbeat. It extends the leases of attempts
// the worker still holds, expires attempts the worker no longer reports
// (after a grace period covering in-flight assignments), and returns the
// reported attempts that are no longer valid so the worker stops them.
func (s *Scheduler) Reconcile(ctx context.Context, projectID, deviceID string, reported []protocol.AttemptStatus) ([]string, error) {
	held := map[string]int64{}
	for _, a := range reported {
		held[a.AttemptID] = a.FencingToken
	}
	var invalid []string
	var changedJobs []string
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		rows, err := tx.Query(ctx, `SELECT a.id, a.fencing_token, a.job_id, a.leased_at, j.lease_seconds,
				(j.current_attempt_id = a.id) AS is_current
			FROM job_attempts a JOIN jobs j ON j.id=a.job_id
			WHERE a.device_id=$1 AND a.state = ANY($2) FOR UPDATE OF a`, deviceID, activeAttemptStates)
		if err != nil {
			return err
		}
		type act struct {
			id, job   string
			fencing   int64
			leasedAt  time.Time
			leaseSecs int
			current   bool
		}
		var acts []act
		for rows.Next() {
			var x act
			if err := rows.Scan(&x.id, &x.fencing, &x.job, &x.leasedAt, &x.leaseSecs, &x.current); err != nil {
				rows.Close()
				return err
			}
			acts = append(acts, x)
		}
		rows.Close()
		if rows.Err() != nil {
			return rows.Err()
		}
		dbActive := map[string]bool{}
		for _, x := range acts {
			f, reportedByWorker := held[x.id]
			switch {
			case reportedByWorker && f == x.fencing && x.current:
				dbActive[x.id] = true
				if _, err := tx.Exec(ctx, `UPDATE job_attempts SET lease_expires_at=now() + make_interval(secs => $2) WHERE id=$1`,
					x.id, x.leaseSecs); err != nil {
					return err
				}
			case !reportedByWorker && time.Since(x.leasedAt) > 30*time.Second:
				// Worker restarted or lost the task: release it now instead of waiting for lease expiry.
				if _, err := tx.Exec(ctx, `UPDATE job_attempts SET state='expired', completed_at=now(), error='not held by worker at heartbeat' WHERE id=$1`, x.id); err != nil {
					return err
				}
				if _, err := tx.Exec(ctx, `UPDATE jobs SET status=CASE WHEN attempt_count < max_attempts THEN 'ready' ELSE 'failed' END,
					current_attempt_id=NULL, updated_at=now(), failure_reason='worker lost attempt'
					WHERE id=$1 AND current_attempt_id=$2`, x.job, x.id); err != nil {
					return err
				}
				changedJobs = append(changedJobs, x.job)
			default:
				dbActive[x.id] = reportedByWorker
			}
		}
		for id := range held {
			if !dbActive[id] {
				invalid = append(invalid, id)
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(changedJobs) > 0 {
		s.d.Events.NotifyJobs(projectID)
		for _, j := range changedJobs {
			s.publishJob(ctx, projectID, j)
		}
	}
	return invalid, nil
}

// publishJob emits a job.updated event with the dashboard view.
func (s *Scheduler) publishJob(ctx context.Context, projectID, jobID string) {
	if s.d.Events.Subscribers() == 0 {
		return
	}
	v, err := s.GetJob(ctx, projectID, jobID)
	if err != nil {
		return
	}
	s.d.Events.Publish(events.Event{Type: "job.updated", ProjectID: projectID, Data: v})
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
