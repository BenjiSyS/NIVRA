// Package training orchestrates distributed DiLoCo-style training of Nivra
// models across workers, versioned global models and checkpoints. All state
// lives in PostgreSQL so a controller restart resumes the run.
package training

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/api"
	"nivra/internal/controller/app"
	"nivra/internal/controller/artifacts"
	"nivra/internal/controller/core"
	"nivra/internal/controller/events"
	"nivra/internal/controller/scheduler"
)

// Errors.
var (
	ErrNotFound  = errors.New("training run not found")
	ErrBadState  = errors.New("operation not allowed in the run's current state")
	ErrBadConfig = errors.New("invalid training configuration")
)

// InnerConfig is passed to ai.local_train (see docs/training/DISTRIBUTED.md).
type InnerConfig struct {
	InnerSteps  int     `json:"inner_steps"`
	BatchSize   int     `json:"batch_size"`
	LR          float64 `json:"lr"`
	WeightDecay float64 `json:"weight_decay"`
	GradClip    float64 `json:"grad_clip"`
	EvalClips   int     `json:"eval_clips"`
}

// RunConfig configures a distributed run.
type RunConfig struct {
	Architecture    string      `json:"architecture"` // ai config name, e.g. nivra_tiny / nivra_small
	Seed            int         `json:"seed"`
	Rounds          int         `json:"rounds"`
	WorkersPerRound int         `json:"workers_per_round"`
	MinUpdates      int         `json:"min_updates"`
	RoundTimeoutSec int         `json:"round_timeout_sec"`
	ShardSize       int         `json:"shard_size"`
	DataSeed        int         `json:"data_seed"`
	OuterLR         float64     `json:"outer_lr"`
	Momentum        float64     `json:"momentum"`
	EvalHeldout     bool        `json:"eval_heldout"`
	Inner           InnerConfig `json:"inner"`
	ReqRAMMB        int         `json:"req_ram_mb"`
	LeaseSeconds    int         `json:"lease_seconds"`
	// InitWeightsArtifact / InitManifestArtifact optionally seed version 0
	// from an existing model bundle (e.g. a single-node trained checkpoint)
	// instead of random initialisation. Both must be set together.
	InitWeightsArtifact string `json:"init_weights_artifact,omitempty"`
	// AllowUntrusted lets any device train; by default only devices an
	// administrator marked trusted receive model-update jobs.
	AllowUntrusted bool `json:"allow_untrusted,omitempty"`
	// MaxHeldoutIncrease rejects an aggregated version whose controller-side
	// held-out loss rises by more than this fraction (requires eval_heldout).
	MaxHeldoutIncrease   float64 `json:"max_heldout_increase,omitempty"`
	InitManifestArtifact string  `json:"init_manifest_artifact,omitempty"`
}

// Defaults fills unset fields with conservative values sized for CPU workers.
func (c *RunConfig) Defaults() {
	if c.Architecture == "" {
		c.Architecture = "nivra_tiny"
	}
	if c.Rounds == 0 {
		c.Rounds = 5
	}
	if c.WorkersPerRound == 0 {
		c.WorkersPerRound = 2
	}
	if c.MinUpdates == 0 {
		c.MinUpdates = c.WorkersPerRound
	}
	if c.RoundTimeoutSec == 0 {
		c.RoundTimeoutSec = 3600
	}
	if c.ShardSize == 0 {
		c.ShardSize = 128
	}
	if c.OuterLR == 0 {
		c.OuterLR = 0.7
	}
	if c.Momentum == 0 {
		c.Momentum = 0.9
	}
	if c.Inner.InnerSteps == 0 {
		c.Inner.InnerSteps = 30
	}
	if c.Inner.BatchSize == 0 {
		c.Inner.BatchSize = 8
	}
	if c.Inner.LR == 0 {
		c.Inner.LR = 3e-4
	}
	if c.Inner.GradClip == 0 {
		c.Inner.GradClip = 1.0
	}
	if c.Inner.EvalClips == 0 {
		c.Inner.EvalClips = 16
	}
	if c.ReqRAMMB == 0 {
		c.ReqRAMMB = 1024
	}
	if c.LeaseSeconds == 0 {
		c.LeaseSeconds = 300
	}
	if c.MaxHeldoutIncrease == 0 {
		c.MaxHeldoutIncrease = 0.10
	}
}

// Validate bounds the configuration.
func (c RunConfig) Validate() error {
	switch {
	case !validName(c.Architecture):
		return fmt.Errorf("%w: architecture must be a config name", ErrBadConfig)
	case c.Rounds < 1 || c.Rounds > 100000:
		return fmt.Errorf("%w: rounds 1..100000", ErrBadConfig)
	case c.WorkersPerRound < 1 || c.WorkersPerRound > 64:
		return fmt.Errorf("%w: workers_per_round 1..64", ErrBadConfig)
	case c.MinUpdates < 1 || c.MinUpdates > c.WorkersPerRound:
		return fmt.Errorf("%w: min_updates 1..workers_per_round", ErrBadConfig)
	case c.RoundTimeoutSec < 60:
		return fmt.Errorf("%w: round_timeout_sec >= 60", ErrBadConfig)
	case c.ShardSize < 8 || c.ShardSize > 4096:
		return fmt.Errorf("%w: shard_size 8..4096", ErrBadConfig)
	case c.OuterLR <= 0 || c.OuterLR > 2 || c.Momentum < 0 || c.Momentum >= 1:
		return fmt.Errorf("%w: outer_lr (0,2], momentum [0,1)", ErrBadConfig)
	case c.Inner.InnerSteps < 1 || c.Inner.InnerSteps > 2000 || c.Inner.BatchSize < 1 || c.Inner.BatchSize > 64:
		return fmt.Errorf("%w: inner_steps 1..2000, batch_size 1..64", ErrBadConfig)
	case c.Inner.LR <= 0 || c.Inner.LR > 1e-2:
		return fmt.Errorf("%w: inner lr (0, 0.01]", ErrBadConfig)
	case c.LeaseSeconds < 30 || c.LeaseSeconds > 86400:
		return fmt.Errorf("%w: lease_seconds 30..86400", ErrBadConfig)
	case (c.InitWeightsArtifact == "") != (c.InitManifestArtifact == ""):
		return fmt.Errorf("%w: init_weights_artifact and init_manifest_artifact must be given together", ErrBadConfig)
	}
	return nil
}

func validName(s string) bool {
	if s == "" || len(s) > 64 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '_' || c == '-') {
			return false
		}
	}
	return true
}

// Service implements training orchestration.
type Service struct {
	d     *core.Deps
	sched *scheduler.Scheduler
	store *artifacts.Store
	kick  chan struct{}
}

// RegisterJobTypes registers the AI job types. svc may be nil for tools that
// only enqueue jobs (validators then reject results).
func RegisterJobTypes(reg *scheduler.Registry, svc *Service) {
	reg.Register(&scheduler.JobType{Name: "ai.local_train", Category: "training", Async: true,
		ValidateParams: objectParams, Validate: svcOr(svc, (*Service).validateLocalTrain), OnFinalized: svcKick(svc)})
	reg.Register(&scheduler.JobType{Name: "ai.dataset_shard", Category: "preprocessing", Async: true,
		ValidateParams: objectParams, Validate: svcOr(svc, (*Service).validateDatasetShard)})
	reg.Register(&scheduler.JobType{Name: "ai.evaluate", Category: "evaluation", Async: true,
		ValidateParams: objectParams, Validate: svcOr(svc, (*Service).validateEvaluate)})
	reg.Register(&scheduler.JobType{Name: "ai.generate", Category: "generation", Async: true,
		ValidateParams: objectParams, Validate: svcOr(svc, (*Service).validateGenerate)})
}

func objectParams(raw json.RawMessage) error {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return errors.New("params must be a JSON object")
	}
	return nil
}

func svcOr(svc *Service, f func(*Service, context.Context, scheduler.ValidationInput) (scheduler.Verdict, error)) func(context.Context, scheduler.ValidationInput) (scheduler.Verdict, error) {
	return func(ctx context.Context, in scheduler.ValidationInput) (scheduler.Verdict, error) {
		if svc == nil {
			return scheduler.Verdict{}, errors.New("training service unavailable")
		}
		return f(svc, ctx, in)
	}
}

func svcKick(svc *Service) func(context.Context, scheduler.ValidationInput, scheduler.Verdict) {
	return func(context.Context, scheduler.ValidationInput, scheduler.Verdict) {
		if svc != nil {
			svc.Kick()
		}
	}
}

// Factory plugs training into the controller (app.ExtensionFactory).
func Factory(rt *app.Runtime) (api.Extension, func(context.Context), error) {
	svc := &Service{d: rt.Deps, sched: rt.Sched, store: rt.Artifacts, kick: make(chan struct{}, 1)}
	RegisterJobTypes(rt.Registry, svc)
	return svc, svc.RunLoop, nil
}

// Kick wakes the orchestrator.
func (s *Service) Kick() {
	select {
	case s.kick <- struct{}{}:
	default:
	}
}

// Run is the dashboard view of a training run.
type Run struct {
	ID           string          `json:"id"`
	ProjectID    string          `json:"project_id"`
	Name         string          `json:"name"`
	Status       string          `json:"status"`
	StatusDetail string          `json:"status_detail"`
	Config       RunConfig       `json:"config"`
	ModelVersion *string         `json:"model_version"`
	VersionNum   *int            `json:"version_number"`
	Round        int             `json:"current_round"`
	Step         int64           `json:"current_step"`
	RoundBatch   int             `json:"round_batch"`
	RoundStarted *time.Time      `json:"round_started_at"`
	VAEArtifact  *string         `json:"vae_artifact"`
	CreatedAt    time.Time       `json:"created_at"`
	UpdatedAt    time.Time       `json:"updated_at"`
	Latest       map[string]any  `json:"latest_metrics"`
	Raw          json.RawMessage `json:"-"`
}

const runSelect = `SELECT r.id, r.project_id, r.name, r.status, r.status_detail, r.config, r.model_version::text, v.version_number,
	r.current_round, r.current_step, r.round_batch, r.round_started_at, r.vae_artifact::text, r.created_at, r.updated_at
	FROM training_runs r LEFT JOIN model_versions v ON v.id = r.model_version `

func scanRun(row pgx.Row) (*Run, error) {
	r := &Run{}
	var cfg []byte
	err := row.Scan(&r.ID, &r.ProjectID, &r.Name, &r.Status, &r.StatusDetail, &cfg, &r.ModelVersion, &r.VersionNum,
		&r.Round, &r.Step, &r.RoundBatch, &r.RoundStarted, &r.VAEArtifact, &r.CreatedAt, &r.UpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	_ = json.Unmarshal(cfg, &r.Config)
	return r, nil
}

// GetRun loads a run.
func (s *Service) GetRun(ctx context.Context, projectID, runID string) (*Run, error) {
	r, err := scanRun(s.d.DB.QueryRow(ctx, runSelect+`WHERE r.id=$1 AND r.project_id=$2`, runID, projectID))
	if err != nil {
		return nil, err
	}
	r.Latest = s.latestMetrics(ctx, r.ID)
	return r, nil
}

func (s *Service) runByID(ctx context.Context, runID string) (*Run, error) {
	return scanRun(s.d.DB.QueryRow(ctx, runSelect+`WHERE r.id=$1`, runID))
}

// ListRuns returns the project's runs, newest first.
func (s *Service) ListRuns(ctx context.Context, projectID string) ([]*Run, error) {
	rows, err := s.d.DB.Query(ctx, runSelect+`WHERE r.project_id=$1 ORDER BY r.created_at DESC LIMIT 100`, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []*Run{}
	for rows.Next() {
		r, err := scanRun(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	if rows.Err() != nil {
		return nil, rows.Err()
	}
	for _, r := range out {
		r.Latest = s.latestMetrics(ctx, r.ID)
	}
	return out, nil
}

func (s *Service) latestMetrics(ctx context.Context, runID string) map[string]any {
	out := map[string]any{}
	rows, err := s.d.DB.Query(ctx, `SELECT DISTINCT ON (name) name, value, round FROM training_metrics WHERE run_id=$1 ORDER BY name, id DESC`, runID)
	if err != nil {
		return out
	}
	defer rows.Close()
	for rows.Next() {
		var n string
		var v float64
		var r int
		if rows.Scan(&n, &v, &r) == nil {
			out[n] = v
		}
	}
	return out
}

// CreateRun creates a run in status "created"; the orchestrator initialises it.
func (s *Service) CreateRun(ctx context.Context, actor core.Actor, projectID, name string, cfg RunConfig, vaeArtifact string) (*Run, error) {
	cfg.Defaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	name = strings.TrimSpace(name)
	if name == "" || len(name) > 80 {
		return nil, fmt.Errorf("%w: name must be 1-80 characters", ErrBadConfig)
	}
	info, err := s.store.Get(ctx, projectID, vaeArtifact)
	if err != nil || info.State != "complete" {
		return nil, fmt.Errorf("%w: vae_artifact must be a complete artifact in this project (upload vae.safetensors first)", ErrBadConfig)
	}
	b, _ := json.Marshal(cfg)
	var by any
	if actor.Type == "user" {
		by = actor.ID
	}
	var id string
	if err := s.d.DB.QueryRow(ctx, `INSERT INTO training_runs (project_id, name, config, status, vae_artifact, dataset_version, created_by)
		VALUES ($1,$2,$3,'created',$4,$5,$6) RETURNING id`, projectID, name, b, vaeArtifact,
		fmt.Sprintf("synthetic-procedural/data_seed=%d/shard=%d", cfg.DataSeed, cfg.ShardSize), by).Scan(&id); err != nil {
		return nil, err
	}
	s.d.Audit(ctx, nil, actor, "training.create", "run:"+id, "success", map[string]any{"name": name})
	s.Kick()
	return s.GetRun(ctx, projectID, id)
}

// transition changes status if the current status is one of from.
func (s *Service) transition(ctx context.Context, actor core.Actor, projectID, runID, to, detail string, from ...string) (*Run, error) {
	tag, err := s.d.DB.Exec(ctx, `UPDATE training_runs SET status=$3, status_detail=$4, updated_at=now()
		WHERE id=$1 AND project_id=$2 AND status = ANY($5)`, runID, projectID, to, detail, from)
	if err != nil {
		return nil, err
	}
	if tag.RowsAffected() == 0 {
		if _, err := s.GetRun(ctx, projectID, runID); err != nil {
			return nil, err
		}
		return nil, ErrBadState
	}
	s.d.Audit(ctx, nil, actor, "training."+to, "run:"+runID, "success", nil)
	s.publishRun(ctx, runID)
	s.Kick()
	return s.GetRun(ctx, projectID, runID)
}

// Pause stops issuing new work; queued jobs are cancelled, running ones finish.
func (s *Service) Pause(ctx context.Context, a core.Actor, pid, rid string) (*Run, error) {
	r, err := s.transition(ctx, a, pid, rid, "paused", "paused by operator", "running", "created")
	if err == nil {
		s.cancelQueued(ctx, rid)
	}
	return r, err
}

// Resume continues a paused run.
func (s *Service) Resume(ctx context.Context, a core.Actor, pid, rid string) (*Run, error) {
	return s.transition(ctx, a, pid, rid, "running", "", "paused")
}

// Stop finishes the run at the next safe boundary (aggregating any valid
// updates of the current round) and keeps the final version as a checkpoint.
func (s *Service) Stop(ctx context.Context, a core.Actor, pid, rid string) (*Run, error) {
	r, err := s.transition(ctx, a, pid, rid, "stopping", "stop requested; finishing at a safe boundary", "running", "paused", "created", "initializing")
	if err == nil {
		s.cancelQueued(ctx, rid)
	}
	return r, err
}

func (s *Service) cancelQueued(ctx context.Context, runID string) {
	rows, err := s.d.DB.Query(ctx, `SELECT id, project_id FROM jobs WHERE training_run_id=$1 AND status='ready'`, runID)
	if err != nil {
		return
	}
	type j struct{ id, p string }
	var js []j
	for rows.Next() {
		var x j
		if rows.Scan(&x.id, &x.p) == nil {
			js = append(js, x)
		}
	}
	rows.Close()
	for _, x := range js {
		_ = s.sched.Cancel(ctx, core.System, x.p, x.id)
	}
}

// SaveCheckpoint marks the current global version as a protected manual checkpoint.
func (s *Service) SaveCheckpoint(ctx context.Context, a core.Actor, pid, rid string) (map[string]any, error) {
	r, err := s.GetRun(ctx, pid, rid)
	if err != nil {
		return nil, err
	}
	if r.ModelVersion == nil {
		return nil, fmt.Errorf("%w: run has no model version yet", ErrBadState)
	}
	var id string
	err = s.d.DB.QueryRow(ctx, `INSERT INTO checkpoints (run_id, project_id, model_version, step, round, trigger, manifest, artifact_id, checksum, metrics, protected)
		SELECT $1, $2, v.id, $3, $4, 'manual', v.manifest, v.weights_artifact, COALESCE(v.manifest->>'weights_sha256',''), $5, true
		FROM model_versions v WHERE v.id=$6 RETURNING id`, rid, pid, r.Step, r.Round, toJSON(r.Latest), *r.ModelVersion).Scan(&id)
	if err != nil {
		return nil, err
	}
	s.d.Audit(ctx, nil, a, "training.checkpoint", "run:"+rid, "success", map[string]any{"checkpoint": id})
	return map[string]any{"checkpoint_id": id, "model_version": *r.ModelVersion}, nil
}

// LoadCheckpoint rolls a paused run back to a checkpoint's model version.
func (s *Service) LoadCheckpoint(ctx context.Context, a core.Actor, pid, rid, checkpointID string) (*Run, error) {
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var status string
		if err := tx.QueryRow(ctx, `SELECT status FROM training_runs WHERE id=$1 AND project_id=$2 FOR UPDATE`, rid, pid).Scan(&status); err != nil {
			return ErrNotFound
		}
		if status != "paused" && status != "stopped" && status != "completed" {
			return fmt.Errorf("%w: pause the run before loading a checkpoint", ErrBadState)
		}
		var mv string
		var round int
		var step int64
		if err := tx.QueryRow(ctx, `SELECT c.model_version, c.round, c.step FROM checkpoints c
			WHERE c.id=$1 AND c.project_id=$2 AND c.state='valid' AND c.model_version IS NOT NULL`, checkpointID, pid).Scan(&mv, &round, &step); err != nil {
			return fmt.Errorf("%w: checkpoint not found or not loadable", ErrBadConfig)
		}
		_, err := tx.Exec(ctx, `UPDATE training_runs SET model_version=$2, current_round=$3, current_step=$4, round_batch=round_batch+1,
			round_started_at=NULL, pending_version=NULL, status='paused', status_detail='checkpoint loaded', updated_at=now() WHERE id=$1`,
			rid, mv, round, step)
		return err
	})
	if err != nil {
		return nil, err
	}
	s.d.Audit(ctx, nil, a, "training.load_checkpoint", "run:"+rid, "success", map[string]any{"checkpoint": checkpointID})
	s.publishRun(ctx, rid)
	return s.GetRun(ctx, pid, rid)
}

func toJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil || string(b) == "null" {
		return []byte(`{}`)
	}
	return b
}

func (s *Service) publishRun(ctx context.Context, runID string) {
	r, err := s.runByID(ctx, runID)
	if err != nil {
		return
	}
	r.Latest = s.latestMetrics(ctx, runID)
	s.d.Events.Publish(events.Event{Type: "training.updated", ProjectID: r.ProjectID, Data: r})
}

// Overview implements api.Extension.
func (s *Service) Overview(ctx context.Context, projectID string) map[string]any {
	out := map[string]any{"training": nil, "latest_checkpoint": nil}
	var rid string
	if err := s.d.DB.QueryRow(ctx, `SELECT id FROM training_runs WHERE project_id=$1 AND status IN ('initializing','running','paused','stopping','created')
		ORDER BY created_at DESC LIMIT 1`, projectID).Scan(&rid); err == nil {
		if r, err := s.GetRun(ctx, projectID, rid); err == nil {
			out["training"] = r
		}
	}
	var cp struct {
		ID      string    `json:"id"`
		Round   int       `json:"round"`
		Step    int64     `json:"step"`
		Trigger string    `json:"trigger"`
		Created time.Time `json:"created_at"`
	}
	if err := s.d.DB.QueryRow(ctx, `SELECT id, round, step, trigger, created_at FROM checkpoints WHERE project_id=$1 AND state='valid'
		ORDER BY created_at DESC LIMIT 1`, projectID).Scan(&cp.ID, &cp.Round, &cp.Step, &cp.Trigger, &cp.Created); err == nil {
		out["latest_checkpoint"] = cp
	}
	return out
}
