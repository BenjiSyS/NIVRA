package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/shared/protocol"
	"nivra/internal/shared/secure"
)

// MaxResultBytes bounds the inline JSON result; large outputs use artifacts.
const MaxResultBytes = 32 << 10

// Complete accepts a result submission. It is idempotent per attempt: a
// duplicate submission returns the existing verdict and is never counted twice.
func (s *Scheduler) Complete(ctx context.Context, deviceID, attemptID string, req protocol.JobCompleteRequest) (protocol.JobCompleteResponse, error) {
	var resp protocol.JobCompleteResponse
	if len(req.Result) == 0 {
		req.Result = json.RawMessage(`{}`)
	}
	if len(req.Result) > MaxResultBytes {
		return resp, fmt.Errorf("result exceeds %d bytes; upload it as an artifact", MaxResultBytes)
	}
	if !json.Valid(req.Result) {
		return resp, errors.New("result is not valid JSON")
	}
	sum := secure.SHA256Hex(req.Result)
	if req.Checksum != "" && req.Checksum != sum {
		return resp, fmt.Errorf("result checksum mismatch")
	}
	var in ValidationInput
	var jt *JobType
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		r, err := lockAttempt(ctx, tx, attemptID)
		if err != nil {
			return err
		}
		if r.deviceID != deviceID {
			return ErrNotYours
		}
		// Duplicate submission for this attempt?
		var prevStatus, prevReason, prevSum string
		err = tx.QueryRow(ctx, `SELECT validation_status, rejection_reason, checksum FROM job_results WHERE attempt_id=$1`, attemptID).
			Scan(&prevStatus, &prevReason, &prevSum)
		if err == nil {
			resp.Duplicate = true
			resp.Reason = prevReason
			switch prevStatus {
			case "accepted":
				resp.Status = protocol.ResultAccepted
			case "rejected":
				resp.Status = protocol.ResultRejected
			default:
				resp.Status = protocol.ResultValidating
			}
			if prevSum != sum {
				resp.Reason = "a different result was already submitted for this attempt; the first submission stands"
			}
			return nil
		}
		if !errors.Is(err, pgx.ErrNoRows) {
			return err
		}
		if err := r.checkCurrent(deviceID, req.FencingToken); err != nil {
			return err
		}
		var ok bool
		if jt, ok = s.reg.Get(r.jobType); !ok {
			return fmt.Errorf("%w: %s", ErrUnknownJobType, r.jobType)
		}
		var artifact *string
		if len(req.Artifacts) > 16 {
			return ErrBadArtifact
		}
		check := func(id string) error {
			var owner *string
			var state, project string
			if err := tx.QueryRow(ctx, `SELECT uploaded_by_device::text, state, project_id::text FROM artifacts WHERE id=$1`, id).
				Scan(&owner, &state, &project); err != nil || state != "complete" || project != r.projectID || owner == nil || *owner != deviceID {
				return ErrBadArtifact
			}
			return nil
		}
		if req.ArtifactID != "" {
			if err := check(req.ArtifactID); err != nil {
				return err
			}
			artifact = &req.ArtifactID
		}
		for kind, id := range req.Artifacts {
			if len(kind) == 0 || len(kind) > 32 {
				return ErrBadArtifact
			}
			if err := check(id); err != nil {
				return err
			}
		}
		arts, _ := json.Marshal(req.Artifacts)
		if req.Artifacts == nil {
			arts = []byte(`{}`)
		}
		var resultID string
		var started *time.Time
		if err := tx.QueryRow(ctx, `INSERT INTO job_results (attempt_id, job_id, artifact_id, artifacts, payload, checksum)
			VALUES ($1, $2, $3, $4, $5, $6) RETURNING id`, attemptID, r.jobID, artifact, arts, req.Result, sum).Scan(&resultID); err != nil {
			return err
		}
		if err := tx.QueryRow(ctx, `UPDATE job_attempts SET state='validating', progress=1, completed_at=now() WHERE id=$1
			RETURNING COALESCE(started_at, leased_at)`, attemptID).Scan(&started); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE jobs SET status='validating', updated_at=now() WHERE id=$1`, r.jobID); err != nil {
			return err
		}
		in = ValidationInput{JobID: r.jobID, AttemptID: attemptID, ResultID: resultID, ProjectID: r.projectID,
			DeviceID: deviceID, JobType: r.jobType, Result: req.Result, ArtifactID: artifact, Artifacts: req.Artifacts}
		if started != nil {
			in.Duration = time.Since(*started)
		}
		return tx.QueryRow(ctx, `SELECT params, model_version::text, training_run_id::text FROM jobs WHERE id=$1`, r.jobID).
			Scan(&in.Params, &in.ModelVersion, &in.TrainingRun)
	})
	if err != nil || resp.Duplicate {
		return resp, err
	}
	s.publishJob(ctx, in.ProjectID, in.JobID)
	if jt.Async {
		s.kickValidation()
		resp.Status = protocol.ResultValidating
		return resp, nil
	}
	v, err := s.runValidator(ctx, jt, in)
	if err != nil {
		// Validator infrastructure failed: leave pending for the background loop.
		s.d.Log.Warn("inline validation failed; deferring", "job", in.JobID, "err", err)
		s.kickValidation()
		resp.Status = protocol.ResultValidating
		return resp, nil
	}
	if err := s.finalize(ctx, jt, in, v); err != nil {
		return resp, err
	}
	resp.Status, resp.Reason = protocol.ResultRejected, v.Reason
	if v.Accept {
		resp.Status = protocol.ResultAccepted
	}
	return resp, nil
}

func (s *Scheduler) runValidator(ctx context.Context, jt *JobType, in ValidationInput) (v Verdict, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("validator panic: %v", r)
		}
	}()
	vctx, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()
	return jt.Validate(vctx, in)
}

// finalize commits a verdict exactly once per result.
func (s *Scheduler) finalize(ctx context.Context, jt *JobType, in ValidationInput, v Verdict) error {
	applied := false
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var status string
		if err := tx.QueryRow(ctx, `SELECT validation_status FROM job_results WHERE id=$1 FOR UPDATE`, in.ResultID).Scan(&status); err != nil {
			return err
		}
		if status != "pending" {
			return nil // already finalised (crash/retry) or cancelled
		}
		r, err := lockAttempt(ctx, tx, in.AttemptID)
		if err != nil {
			return err
		}
		detail, _ := json.Marshal(v.Detail)
		if detail == nil || string(detail) == "null" {
			detail = []byte(`{}`)
		}
		if v.Accept {
			if _, err := tx.Exec(ctx, `UPDATE job_results SET validation_status='accepted', validated_at=now(), accepted_at=now(),
				validation_detail=$2 WHERE id=$1`, in.ResultID, detail); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `UPDATE job_attempts SET state='accepted' WHERE id=$1`, in.AttemptID); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `UPDATE jobs SET status='completed', completed_at=now(), updated_at=now() WHERE id=$1`, in.JobID); err != nil {
				return err
			}
			var category string
			if err := tx.QueryRow(ctx, `SELECT category FROM jobs WHERE id=$1`, in.JobID).Scan(&category); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO contributions (project_id, device_id, job_id, attempt_id, category,
					measured_seconds, units, unit_kind, credit)
				SELECT $1, $2, $3, $4, $5, EXTRACT(EPOCH FROM (a.completed_at - COALESCE(a.started_at, a.leased_at))), $6, $7, $8
				FROM job_attempts a WHERE a.id=$4
				ON CONFLICT (attempt_id) DO NOTHING`, in.ProjectID, in.DeviceID, in.JobID, in.AttemptID, category,
				v.Units, v.UnitKind, v.Credit); err != nil {
				return err
			}
		} else {
			if _, err := tx.Exec(ctx, `UPDATE job_results SET validation_status='rejected', validated_at=now(), rejection_reason=$2,
				validation_detail=$3 WHERE id=$1`, in.ResultID, truncate(v.Reason, 1000), detail); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `UPDATE job_attempts SET state='rejected', error=$2 WHERE id=$1`, in.AttemptID, truncate(v.Reason, 1000)); err != nil {
				return err
			}
			next := JobFailed
			if v.Retry && r.attemptCount < r.maxAttempts {
				next = JobReady
			}
			if _, err := tx.Exec(ctx, `UPDATE jobs SET status=$2, current_attempt_id=NULL, failure_reason=$3, updated_at=now(),
				completed_at=CASE WHEN $2='failed' THEN now() ELSE NULL END WHERE id=$1 AND status='validating'`,
				in.JobID, next, "result rejected: "+truncate(v.Reason, 900)); err != nil {
				return err
			}
		}
		applied = true
		return nil
	})
	if err != nil || !applied {
		return err
	}
	if jt.OnFinalized != nil {
		jt.OnFinalized(ctx, in, v)
	}
	if !v.Accept {
		s.d.Events.NotifyJobs(in.ProjectID)
	}
	s.publishJob(ctx, in.ProjectID, in.JobID)
	s.d.Events.Publish(eventsResult(in, v))
	return nil
}

func (s *Scheduler) kickValidation() {
	select {
	case s.valCh <- struct{}{}:
	default:
	}
}

// RunValidationLoop validates pending results (async job types and any left
// pending by a crash) until ctx is cancelled.
func (s *Scheduler) RunValidationLoop(ctx context.Context) {
	t := time.NewTicker(10 * time.Second)
	defer t.Stop()
	for {
		s.validatePending(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		case <-s.valCh:
		}
	}
}

func (s *Scheduler) validatePending(ctx context.Context) {
	rows, err := s.d.DB.Query(ctx, `SELECT r.id, r.attempt_id, r.job_id, j.project_id, a.device_id, j.job_type, j.params, r.payload,
			r.artifact_id::text, r.artifacts, j.model_version::text, j.training_run_id::text,
			EXTRACT(EPOCH FROM (a.completed_at - COALESCE(a.started_at, a.leased_at)))
		FROM job_results r JOIN job_attempts a ON a.id=r.attempt_id JOIN jobs j ON j.id=r.job_id
		WHERE r.validation_status='pending' ORDER BY r.submitted_at LIMIT 20`)
	if err != nil {
		if ctx.Err() == nil {
			s.d.Log.Warn("list pending results", "err", err)
		}
		return
	}
	var pending []ValidationInput
	for rows.Next() {
		var in ValidationInput
		var secs *float64
		if err := rows.Scan(&in.ResultID, &in.AttemptID, &in.JobID, &in.ProjectID, &in.DeviceID, &in.JobType, &in.Params,
			&in.Result, &in.ArtifactID, &in.Artifacts, &in.ModelVersion, &in.TrainingRun, &secs); err != nil {
			s.d.Log.Warn("scan pending result", "err", err)
			continue
		}
		if secs != nil {
			in.Duration = time.Duration(*secs * float64(time.Second))
		}
		pending = append(pending, in)
	}
	rows.Close()
	for _, in := range pending {
		if ctx.Err() != nil {
			return
		}
		jt, ok := s.reg.Get(in.JobType)
		if !ok {
			continue
		}
		v, err := s.runValidator(ctx, jt, in)
		if err != nil {
			s.d.Log.Warn("validator error; will retry", "job", in.JobID, "type", in.JobType, "err", err)
			continue
		}
		if err := s.finalize(ctx, jt, in, v); err != nil {
			s.d.Log.Warn("finalize failed", "job", in.JobID, "err", err)
		}
	}
}
