package scheduler

import (
	"context"
	"encoding/json"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/events"
)

// AttemptView is the dashboard view of an attempt.
type AttemptView struct {
	ID             string     `json:"id"`
	DeviceID       string     `json:"device_id"`
	DeviceName     string     `json:"device_name"`
	AttemptNumber  int        `json:"attempt_number"`
	FencingToken   int64      `json:"fencing_token"`
	State          string     `json:"state"`
	Progress       float64    `json:"progress"`
	Stage          string     `json:"stage"`
	LeaseExpiresAt time.Time  `json:"lease_expires_at"`
	LeasedAt       time.Time  `json:"leased_at"`
	StartedAt      *time.Time `json:"started_at"`
	CompletedAt    *time.Time `json:"completed_at"`
	Error          string     `json:"error"`
}

// JobView is the dashboard view of a job.
type JobView struct {
	ID            string          `json:"id"`
	JobType       string          `json:"job_type"`
	Category      string          `json:"category"`
	Status        string          `json:"status"`
	Priority      int             `json:"priority"`
	AttemptCount  int             `json:"attempt_count"`
	MaxAttempts   int             `json:"max_attempts"`
	TrainingRunID *string         `json:"training_run_id"`
	Params        json.RawMessage `json:"params"`
	FailureReason string          `json:"failure_reason"`
	CreatedAt     time.Time       `json:"created_at"`
	UpdatedAt     time.Time       `json:"updated_at"`
	CompletedAt   *time.Time      `json:"completed_at"`
	Current       *AttemptView    `json:"current_attempt"`
	Result        *ResultView     `json:"result,omitempty"`
}

// ResultView summarises the latest result of a job.
type ResultView struct {
	Status      string          `json:"validation_status"`
	Reason      string          `json:"rejection_reason"`
	Payload     json.RawMessage `json:"payload"`
	ArtifactID  *string         `json:"artifact_id"`
	SubmittedAt time.Time       `json:"submitted_at"`
}

const jobSelect = `SELECT j.id, j.job_type, j.category, j.status, j.priority, j.attempt_count, j.max_attempts,
	j.training_run_id::text, j.params, j.failure_reason, j.created_at, j.updated_at, j.completed_at,
	a.id::text, a.device_id::text, d.name, a.attempt_number, a.fencing_token, a.state, a.progress, a.stage,
	a.lease_expires_at, a.leased_at, a.started_at, a.completed_at, a.error
	FROM jobs j
	LEFT JOIN LATERAL (SELECT * FROM job_attempts x WHERE x.job_id=j.id ORDER BY x.attempt_number DESC LIMIT 1) a ON true
	LEFT JOIN devices d ON d.id = a.device_id `

func scanJob(r pgx.Row) (JobView, error) {
	var v JobView
	var aID, aDev, aName, aState, aStage, aErr *string
	var aNum *int
	var aFence *int64
	var aProg *float64
	var aExp, aLeased *time.Time
	var aStart, aDone *time.Time
	err := r.Scan(&v.ID, &v.JobType, &v.Category, &v.Status, &v.Priority, &v.AttemptCount, &v.MaxAttempts,
		&v.TrainingRunID, &v.Params, &v.FailureReason, &v.CreatedAt, &v.UpdatedAt, &v.CompletedAt,
		&aID, &aDev, &aName, &aNum, &aFence, &aState, &aProg, &aStage, &aExp, &aLeased, &aStart, &aDone, &aErr)
	if err != nil {
		return v, err
	}
	if aID != nil {
		v.Current = &AttemptView{ID: *aID, DeviceID: deref(aDev), DeviceName: deref(aName), AttemptNumber: *aNum,
			FencingToken: *aFence, State: deref(aState), Progress: *aProg, Stage: deref(aStage),
			LeaseExpiresAt: *aExp, LeasedAt: *aLeased, StartedAt: aStart, CompletedAt: aDone, Error: deref(aErr)}
	}
	return v, nil
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// GetJob returns one job with its latest attempt and result.
func (s *Scheduler) GetJob(ctx context.Context, projectID, jobID string) (JobView, error) {
	v, err := scanJob(s.d.DB.QueryRow(ctx, jobSelect+`WHERE j.id=$1 AND j.project_id=$2`, jobID, projectID))
	if err != nil {
		return v, err
	}
	var rv ResultView
	err = s.d.DB.QueryRow(ctx, `SELECT validation_status, rejection_reason, payload, artifact_id::text, submitted_at
		FROM job_results WHERE job_id=$1 ORDER BY submitted_at DESC LIMIT 1`, jobID).
		Scan(&rv.Status, &rv.Reason, &rv.Payload, &rv.ArtifactID, &rv.SubmittedAt)
	if err == nil {
		v.Result = &rv
	}
	return v, nil
}

// JobFilter selects jobs for listing.
type JobFilter struct {
	Status  string
	JobType string
	RunID   string
	Before  *time.Time
	Limit   int
}

// ListJobs returns jobs newest first with keyset pagination on created_at.
func (s *Scheduler) ListJobs(ctx context.Context, projectID string, f JobFilter) ([]JobView, error) {
	if f.Limit <= 0 || f.Limit > 500 {
		f.Limit = 100
	}
	rows, err := s.d.DB.Query(ctx, jobSelect+`WHERE j.project_id=$1
		AND ($2 = '' OR j.status = $2) AND ($3 = '' OR j.job_type = $3)
		AND ($4 = '' OR j.training_run_id::text = $4)
		AND ($5::timestamptz IS NULL OR j.created_at < $5)
		ORDER BY j.created_at DESC LIMIT $6`, projectID, f.Status, f.JobType, f.RunID, f.Before, f.Limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []JobView
	for rows.Next() {
		v, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, rows.Err()
}

// QueueCounts returns job counts by status.
func (s *Scheduler) QueueCounts(ctx context.Context, projectID string) (map[string]int, error) {
	rows, err := s.d.DB.Query(ctx, `SELECT status, count(*) FROM jobs WHERE project_id=$1 GROUP BY status`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]int{}
	for rows.Next() {
		var st string
		var n int
		if err := rows.Scan(&st, &n); err != nil {
			return nil, err
		}
		out[st] = n
	}
	return out, rows.Err()
}

func eventsResult(in ValidationInput, v Verdict) events.Event {
	return events.Event{Type: "result.finalized", ProjectID: in.ProjectID, Data: map[string]any{
		"job_id": in.JobID, "attempt_id": in.AttemptID, "device_id": in.DeviceID, "job_type": in.JobType,
		"accepted": v.Accept, "reason": v.Reason, "credit": v.Credit, "units": v.Units, "unit_kind": v.UnitKind,
	}}
}
