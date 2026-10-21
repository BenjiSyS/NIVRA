package training

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/controller/scheduler"
	"nivra/internal/shared/protocol"
	"nivra/internal/shared/secure"
)

// version is a global model version with its artifacts.
type version struct {
	ID        string
	VersionID string // manifest version_id (equals ID for controller-created versions)
	Number    int
	Weights   string
	Manifest  string
	State     *string
}

func (s *Service) version(ctx context.Context, id string) (*version, error) {
	v := &version{ID: id}
	var man []byte
	err := s.d.DB.QueryRow(ctx, `SELECT version_number, weights_artifact::text, manifest_artifact::text, state_artifact::text, manifest
		FROM model_versions WHERE id=$1`, id).Scan(&v.Number, &v.Weights, &v.Manifest, &v.State, &man)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, errNoVersion
	}
	if err != nil {
		return nil, err
	}
	var m struct {
		VersionID string `json:"version_id"`
	}
	_ = json.Unmarshal(man, &m)
	v.VersionID = m.VersionID
	return v, nil
}

func roundKeyPrefix(runID string, round int) string { return fmt.Sprintf("run:%s:r%d:", runID, round) }

// RunLoop drives all active runs until ctx is cancelled.
func (s *Service) RunLoop(ctx context.Context) {
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		s.tick(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		case <-s.kick:
		}
	}
}

func (s *Service) tick(ctx context.Context) {
	rows, err := s.d.DB.Query(ctx, `SELECT id FROM training_runs WHERE status IN ('created','initializing','running','stopping') ORDER BY created_at`)
	if err != nil {
		return
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		return
	}
	for _, id := range ids {
		if ctx.Err() != nil {
			return
		}
		run, err := s.runByID(ctx, id)
		if err != nil {
			continue
		}
		if err := s.step(ctx, run); err != nil && ctx.Err() == nil {
			s.d.Log.Error("training step failed", "run_id", run.ID, "project_id", run.ProjectID, "err", err)
			s.setDetail(ctx, run.ID, "error: "+truncate(err.Error(), 300))
		}
	}
}

func (s *Service) setDetail(ctx context.Context, runID, detail string) {
	_, _ = s.d.DB.Exec(ctx, `UPDATE training_runs SET status_detail=$2, updated_at=now() WHERE id=$1`, runID, detail)
	s.publishRun(ctx, runID)
}

func (s *Service) setStatus(ctx context.Context, runID, status, detail string) {
	_, _ = s.d.DB.Exec(ctx, `UPDATE training_runs SET status=$2, status_detail=$3, updated_at=now() WHERE id=$1`, runID, status, detail)
	s.d.Log.Info("training run "+status, "run_id", runID, "detail", detail)
	s.publishRun(ctx, runID)
}

func (s *Service) step(ctx context.Context, run *Run) error {
	switch run.Status {
	case "created", "initializing":
		return s.initRun(ctx, run)
	case "running":
		if run.Round >= run.Config.Rounds {
			s.finalCheckpoint(ctx, run, "stop")
			s.setStatus(ctx, run.ID, "completed", fmt.Sprintf("completed %d rounds", run.Round))
			return nil
		}
		if err := s.ensureRoundJobs(ctx, run); err != nil {
			return err
		}
		return s.maybeAggregate(ctx, run, false)
	case "stopping":
		if err := s.maybeAggregate(ctx, run, true); err != nil {
			return err
		}
		s.cancelRound(ctx, run)
		s.finalCheckpoint(ctx, run, "stop")
		s.setStatus(ctx, run.ID, "stopped", fmt.Sprintf("stopped after round %d", run.Round))
	}
	return nil
}

// initRun creates global version 0 with ai.init_model on the controller.
func (s *Service) initRun(ctx context.Context, run *Run) error {
	if run.ModelVersion != nil {
		s.setStatus(ctx, run.ID, "running", "")
		return nil
	}
	_, _ = s.d.DB.Exec(ctx, `UPDATE training_runs SET status='initializing', status_detail='creating global version 0', updated_at=now(),
		pending_version=COALESCE(pending_version, gen_random_uuid()) WHERE id=$1`, run.ID)
	var pending string
	if err := s.d.DB.QueryRow(ctx, `SELECT pending_version::text FROM training_runs WHERE id=$1`, run.ID).Scan(&pending); err != nil {
		return err
	}
	if run.VAEArtifact == nil {
		s.setStatus(ctx, run.ID, "failed", "no VAE artifact")
		return nil
	}
	vaePath, vaeInfo, err := s.store.Path(ctx, run.ProjectID, *run.VAEArtifact)
	if err != nil {
		s.setStatus(ctx, run.ID, "failed", "VAE artifact unavailable")
		return nil
	}
	if run.Config.InitWeightsArtifact != "" {
		return s.initFromBundle(ctx, run, pending, vaeInfo.SHA256)
	}
	res, _, cleanup, err := s.runPython(ctx, "ai.init_model", run.ID,
		map[string]any{"config": run.Config.Architecture, "seed": run.Config.Seed, "version_id": pending},
		map[string]any{"vae_checkpoint": vaePath})
	if err != nil {
		s.setStatus(ctx, run.ID, "failed", "ai.init_model failed: "+truncate(err.Error(), 400))
		return nil
	}
	defer cleanup()
	w, ok1 := artifactByKind(res, "model_weights")
	m, ok2 := artifactByKind(res, "model_version")
	if !ok1 || !ok2 {
		s.setStatus(ctx, run.ID, "failed", "ai.init_model produced no model_version/model_weights artifacts")
		return nil
	}
	wi, err := s.store.Import(ctx, run.ProjectID, "model_weights", w.Path, "application/octet-stream")
	if err != nil {
		return err
	}
	mi, err := s.store.Import(ctx, run.ProjectID, "model_version", m.Path, "application/json")
	if err != nil {
		return err
	}
	man, err := os.ReadFile(m.Path)
	if err != nil {
		return err
	}
	var sum struct {
		ParamCount       int64  `json:"param_count"`
		TokenizerVersion string `json:"tokenizer_version"`
		Architecture     string `json:"architecture_version"`
	}
	_ = json.Unmarshal(res.Summary, &sum)
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `INSERT INTO model_architectures (id, source_revision, config, tokenizer_version, parameter_count)
			VALUES ($1, $2, $3, $4, $5) ON CONFLICT (id) DO UPDATE SET parameter_count=EXCLUDED.parameter_count`,
			run.Config.Architecture, sum.Architecture, toJSON(map[string]any{"architecture_version": sum.Architecture}),
			sum.TokenizerVersion, sum.ParamCount); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `INSERT INTO model_versions (id, project_id, run_id, architecture_id, version_number, weights_artifact, manifest_artifact, manifest)
			VALUES ($1,$2,$3,$4,0,$5,$6,$7) ON CONFLICT (id) DO NOTHING`, pending, run.ProjectID, run.ID, run.Config.Architecture, wi.ID, mi.ID, man); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE training_runs SET model_version=$2, architecture_id=$3, pending_version=NULL, status='running',
			status_detail='', current_round=0, round_batch=0, round_started_at=NULL, updated_at=now() WHERE id=$1`,
			run.ID, pending, run.Config.Architecture); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE projects SET active_model_version=$2, updated_at=now() WHERE id=$1`, run.ProjectID, pending); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `INSERT INTO checkpoints (run_id, project_id, model_version, step, round, trigger, manifest, artifact_id, checksum, protected)
			VALUES ($1,$2,$3,0,0,'init',$4,$5,$6,true)`, run.ID, run.ProjectID, pending, man, wi.ID, wi.SHA256)
		return err
	})
	if err != nil {
		return err
	}
	s.d.Log.Info("training run initialised", "run_id", run.ID, "project_id", run.ProjectID, "params", sum.ParamCount)
	s.publishRun(ctx, run.ID)
	s.Kick()
	return nil
}

// roundJobs summarises the jobs of the run's current round.
type roundJobs struct {
	batchJobs  int // jobs created in the current batch
	active     int // non-terminal jobs in the round
	accepted   []acceptedUpdate
	terminalNo int // terminal jobs in the current batch without an accepted result
}

type acceptedUpdate struct {
	AttemptID string
	JobID     string
	DeviceID  string
	UpdateID  string
	Artifacts map[string]string
	Detail    map[string]any
}

func (s *Service) loadRound(ctx context.Context, run *Run) (*roundJobs, error) {
	prefix := roundKeyPrefix(run.ID, run.Round)
	batch := fmt.Sprintf("%sb%d:", prefix, run.RoundBatch)
	rj := &roundJobs{}
	err := s.d.DB.QueryRow(ctx, `SELECT
			count(*) FILTER (WHERE idempotency_key LIKE $2),
			count(*) FILTER (WHERE status NOT IN ('completed','failed','cancelled')),
			count(*) FILTER (WHERE idempotency_key LIKE $2 AND status IN ('failed','cancelled'))
		FROM jobs WHERE training_run_id=$3 AND idempotency_key LIKE $1`, prefix+"%", batch+"%", run.ID).
		Scan(&rj.batchJobs, &rj.active, &rj.terminalNo)
	if err != nil {
		return nil, err
	}
	rows, err := s.d.DB.Query(ctx, `SELECT r.attempt_id, r.job_id, a.device_id, r.artifacts, r.validation_detail
		FROM job_results r JOIN jobs j ON j.id=r.job_id JOIN job_attempts a ON a.id=r.attempt_id
		WHERE j.training_run_id=$1 AND j.idempotency_key LIKE $2 AND r.validation_status='accepted' AND j.model_version=$3
		ORDER BY r.accepted_at`, run.ID, prefix+"%", *run.ModelVersion)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var u acceptedUpdate
		var arts, det []byte
		if err := rows.Scan(&u.AttemptID, &u.JobID, &u.DeviceID, &arts, &det); err != nil {
			return nil, err
		}
		_ = json.Unmarshal(arts, &u.Artifacts)
		_ = json.Unmarshal(det, &u.Detail)
		if id, ok := u.Detail["update_id"].(string); ok {
			u.UpdateID = id
		}
		rj.accepted = append(rj.accepted, u)
	}
	return rj, rows.Err()
}

// ensureRoundJobs creates the round's local_train jobs (idempotently) and
// opens a new batch when every job of the batch ended without enough updates.
func (s *Service) ensureRoundJobs(ctx context.Context, run *Run) error {
	rj, err := s.loadRound(ctx, run)
	if err != nil {
		return err
	}
	if rj.batchJobs > 0 {
		if rj.active == 0 && len(rj.accepted) < run.Config.MinUpdates {
			if run.RoundBatch >= 20 {
				s.setStatus(ctx, run.ID, "paused", fmt.Sprintf("round %d failed 20 batches without enough valid updates; inspect logs", run.Round))
				return nil
			}
			_, err := s.d.DB.Exec(ctx, `UPDATE training_runs SET round_batch=round_batch+1, updated_at=now() WHERE id=$1`, run.ID)
			s.d.Log.Warn("round batch ended without enough valid updates; retrying", "run_id", run.ID, "round", run.Round)
			s.Kick()
			return err
		}
		return nil
	}
	base, err := s.version(ctx, *run.ModelVersion)
	if err != nil {
		return err
	}
	need := run.Config.WorkersPerRound - len(rj.accepted)
	if need <= 0 {
		return nil
	}
	c := run.Config
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		for i := 0; i < need; i++ {
			slot := run.Round*c.WorkersPerRound*8 + run.RoundBatch*c.WorkersPerRound + i
			params := map[string]any{
				"update_id":    newUUID(),
				"shard":        map[string]any{"seed": c.DataSeed, "start": slot * c.ShardSize, "count": c.ShardSize},
				"inner_steps":  c.Inner.InnerSteps,
				"batch_size":   c.Inner.BatchSize,
				"lr":           c.Inner.LR,
				"weight_decay": c.Inner.WeightDecay,
				"grad_clip":    c.Inner.GradClip,
				"seed":         c.Seed*1000003 + slot,
				"eval_clips":   c.Inner.EvalClips,
			}
			pb, _ := json.Marshal(params)
			mv, rid := *run.ModelVersion, run.ID
			_, err := s.sched.Create(ctx, tx, scheduler.NewJob{ProjectID: run.ProjectID, JobType: "ai.local_train", Params: pb,
				Inputs: []protocol.ArtifactRef{{ID: base.Weights, Role: "base_weights"}, {ID: base.Manifest, Role: "base_manifest"},
					{ID: *run.VAEArtifact, Role: "vae_weights"}},
				ReqRAMMB: c.ReqRAMMB, Priority: 10, MaxAttempts: 3, LeaseSeconds: c.LeaseSeconds, ReqTrusted: !c.AllowUntrusted,
				IdempotencyKey: fmt.Sprintf("%sb%d:w%d", roundKeyPrefix(run.ID, run.Round), run.RoundBatch, i),
				ModelVersion:   &mv, TrainingRunID: &rid})
			if err != nil {
				return err
			}
		}
		_, err := tx.Exec(ctx, `UPDATE training_runs SET round_started_at=COALESCE(round_started_at, now()), status_detail=$2, updated_at=now() WHERE id=$1`,
			run.ID, fmt.Sprintf("round %d: waiting for %d worker update(s)", run.Round, need))
		return err
	})
	if err != nil {
		return err
	}
	s.d.Events.NotifyJobs(run.ProjectID)
	s.d.Log.Info("training round jobs created", "run_id", run.ID, "project_id", run.ProjectID, "round", run.Round, "jobs", need)
	s.publishRun(ctx, run.ID)
	return nil
}

// maybeAggregate combines the round's valid updates into a new global
// version once enough have arrived (or, when force is set, whatever exists).
func (s *Service) maybeAggregate(ctx context.Context, run *Run, force bool) error {
	if run.ModelVersion == nil {
		return nil
	}
	rj, err := s.loadRound(ctx, run)
	if err != nil {
		return err
	}
	n := len(rj.accepted)
	if n == 0 {
		return nil
	}
	timedOut := run.RoundStarted != nil && time.Since(*run.RoundStarted) > time.Duration(run.Config.RoundTimeoutSec)*time.Second
	ready := force || n >= run.Config.WorkersPerRound ||
		(n >= run.Config.MinUpdates && (rj.active == 0 || timedOut))
	if !ready {
		return nil
	}
	return s.aggregate(ctx, run, rj.accepted)
}

func (s *Service) aggregate(ctx context.Context, run *Run, ups []acceptedUpdate) error {
	start := time.Now()
	if _, err := s.d.DB.Exec(ctx, `UPDATE training_runs SET pending_version=COALESCE(pending_version, gen_random_uuid()), status_detail=$2
		WHERE id=$1`, run.ID, fmt.Sprintf("round %d: aggregating %d update(s)", run.Round, len(ups))); err != nil {
		return err
	}
	var pending string
	if err := s.d.DB.QueryRow(ctx, `SELECT pending_version::text FROM training_runs WHERE id=$1`, run.ID).Scan(&pending); err != nil {
		return err
	}
	s.publishRun(ctx, run.ID)
	base, err := s.version(ctx, *run.ModelVersion)
	if err != nil {
		return err
	}
	path := func(id string) (string, error) {
		p, _, err := s.store.Path(ctx, run.ProjectID, id)
		return p, err
	}
	bw, err := path(base.Weights)
	if err != nil {
		return err
	}
	bm, err := path(base.Manifest)
	if err != nil {
		return err
	}
	inputs := map[string]any{"base_weights": bw, "base_manifest": bm}
	if base.State != nil {
		if p, err := path(*base.State); err == nil {
			inputs["outer_state"] = p
		}
	}
	var list []map[string]string
	for _, u := range ups {
		wp, err1 := path(u.Artifacts["update_weights"])
		mp, err2 := path(u.Artifacts["update"])
		if err1 != nil || err2 != nil {
			continue
		}
		list = append(list, map[string]string{"path": wp, "manifest_path": mp})
	}
	if len(list) == 0 {
		return errors.New("no readable update artifacts")
	}
	inputs["updates"] = list
	if run.Config.EvalHeldout {
		if p, err := path(*run.VAEArtifact); err == nil {
			inputs["vae_weights"] = p
		}
	}
	res, _, cleanup, err := s.runPython(ctx, "ai.aggregate", run.ID, map[string]any{
		"outer_lr": run.Config.OuterLR, "momentum": run.Config.Momentum, "nesterov": true,
		"version_id": pending, "eval_heldout": run.Config.EvalHeldout && inputs["vae_weights"] != nil}, inputs)
	if err != nil {
		return err
	}
	defer cleanup()
	var sum struct {
		NewVersionID *string  `json:"new_version_id"`
		NoOp         bool     `json:"no_op"`
		Applied      []string `json:"applied_update_ids"`
		Rejected     []struct {
			UpdateID string `json:"update_id"`
			Reason   string `json:"reason"`
		} `json:"rejected"`
		AggregatedCount int      `json:"aggregated_count"`
		Method          string   `json:"aggregation_method"`
		DeltaL2         float64  `json:"delta_l2"`
		HeldoutBefore   *float64 `json:"heldout_loss_before"`
		HeldoutAfter    *float64 `json:"heldout_loss_after"`
	}
	if err := json.Unmarshal(res.Summary, &sum); err != nil {
		return fmt.Errorf("aggregate summary: %w", err)
	}
	rejected := map[string]string{}
	for _, r := range sum.Rejected {
		rejected[r.UpdateID] = r.Reason
	}
	applied := map[string]bool{}
	for _, id := range sum.Applied {
		applied[id] = true
	}
	if sum.NoOp || sum.NewVersionID == nil {
		// Every update was rejected at aggregation: record and retry the round.
		_ = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
			for _, u := range ups {
				if err := s.recordUpdate(ctx, tx, run, base, u, "invalid", rejected[u.UpdateID], nil); err != nil {
					return err
				}
			}
			_, err := tx.Exec(ctx, `UPDATE training_runs SET round_batch=round_batch+1, pending_version=NULL WHERE id=$1`, run.ID)
			return err
		})
		s.d.Log.Warn("aggregation rejected every update; retrying round", "run_id", run.ID, "round", run.Round)
		return nil
	}
	// Independent gate: refuse a version whose held-out loss (computed here,
	// on the controller) gets clearly worse; the round is retried instead.
	if sum.HeldoutBefore != nil && sum.HeldoutAfter != nil && run.Config.MaxHeldoutIncrease > 0 &&
		*sum.HeldoutAfter > *sum.HeldoutBefore*(1+run.Config.MaxHeldoutIncrease) {
		why := fmt.Sprintf("aggregated version rejected: held-out loss %.4f -> %.4f", *sum.HeldoutBefore, *sum.HeldoutAfter)
		_ = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
			for _, u := range ups {
				if err := s.recordUpdate(ctx, tx, run, base, u, "invalid", why, nil); err != nil {
					return err
				}
			}
			_, err := tx.Exec(ctx, `UPDATE training_runs SET round_batch=round_batch+1, pending_version=NULL, status_detail=$2 WHERE id=$1`, run.ID, why)
			return err
		})
		s.d.Log.Warn(why, "run_id", run.ID, "round", run.Round)
		return nil
	}
	w, ok1 := artifactByKind(res, "model_weights")
	m, ok2 := artifactByKind(res, "model_version")
	if !ok1 || !ok2 {
		return errors.New("ai.aggregate produced no model artifacts")
	}
	wi, err := s.store.Import(ctx, run.ProjectID, "model_weights", w.Path, "application/octet-stream")
	if err != nil {
		return err
	}
	mi, err := s.store.Import(ctx, run.ProjectID, "model_version", m.Path, "application/json")
	if err != nil {
		return err
	}
	var stateID *string
	if st, ok := artifactByKind(res, "outer_state"); ok {
		si, err := s.store.Import(ctx, run.ProjectID, "outer_state", st.Path, "application/octet-stream")
		if err != nil {
			return err
		}
		stateID = &si.ID
	}
	man, err := os.ReadFile(m.Path)
	if err != nil {
		return err
	}
	var lossFirst, lossLast float64
	devices := map[string]bool{}
	for _, u := range ups {
		lossFirst += num(u.Detail["loss_first"])
		lossLast += num(u.Detail["loss_last"])
		devices[u.DeviceID] = true
	}
	nu := float64(len(ups))
	newRound := run.Round + 1
	newStep := run.Step + int64(run.Config.Inner.InnerSteps)
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `INSERT INTO model_versions (id, project_id, run_id, architecture_id, version_number, parent_version,
				weights_artifact, manifest_artifact, state_artifact, manifest)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) ON CONFLICT (id) DO NOTHING`,
			pending, run.ProjectID, run.ID, run.Config.Architecture, base.Number+1, base.ID, wi.ID, mi.ID, stateID, man); err != nil {
			return err
		}
		for _, u := range ups {
			state, why := "applied", ""
			if !applied[u.UpdateID] {
				state, why = "invalid", rejected[u.UpdateID]
				if why == "" {
					why = "not applied by aggregator"
				}
			}
			var res *string
			if state == "applied" {
				res = &pending
			}
			if err := s.recordUpdate(ctx, tx, run, base, u, state, why, res); err != nil {
				return err
			}
		}
		if _, err := tx.Exec(ctx, `UPDATE training_runs SET model_version=$2, current_round=$3, current_step=$4, round_batch=0,
			round_started_at=NULL, pending_version=NULL, status_detail=$5, updated_at=now() WHERE id=$1`,
			run.ID, pending, newRound, newStep, fmt.Sprintf("round %d aggregated (%d update(s))", run.Round, sum.AggregatedCount)); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE projects SET active_model_version=$2, updated_at=now() WHERE id=$1`, run.ProjectID, pending); err != nil {
			return err
		}
		metrics := map[string]any{"delta_l2": sum.DeltaL2, "aggregated": sum.AggregatedCount, "method": sum.Method}
		if sum.HeldoutAfter != nil {
			metrics["heldout_loss_after"] = *sum.HeldoutAfter
		}
		if _, err := tx.Exec(ctx, `INSERT INTO checkpoints (run_id, project_id, model_version, step, round, trigger, manifest, artifact_id, checksum, metrics)
			VALUES ($1,$2,$3,$4,$5,'round',$6,$7,$8,$9)`, run.ID, run.ProjectID, pending, newStep, newRound, man, wi.ID, wi.SHA256, toJSON(metrics)); err != nil {
			return err
		}
		ins := func(name string, v float64) error {
			_, err := tx.Exec(ctx, `INSERT INTO training_metrics (run_id, round, step, name, value) VALUES ($1,$2,$3,$4,$5)`, run.ID, newRound, newStep, name, v)
			return err
		}
		vals := map[string]float64{"accepted_updates": float64(sum.AggregatedCount), "rejected_updates": float64(len(sum.Rejected)),
			"delta_l2": sum.DeltaL2, "update_loss_first_mean": lossFirst / nu, "update_loss_last_mean": lossLast / nu,
			"round_seconds": time.Since(start).Seconds(), "active_workers": float64(len(devices)), "outer_lr": run.Config.OuterLR}
		if run.RoundStarted != nil {
			vals["round_seconds"] = time.Since(*run.RoundStarted).Seconds()
		}
		if sum.HeldoutBefore != nil {
			vals["heldout_loss_before"] = *sum.HeldoutBefore
		}
		if sum.HeldoutAfter != nil {
			vals["heldout_loss_after"] = *sum.HeldoutAfter
		}
		for k, v := range vals {
			if err := ins(k, v); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	s.d.Log.Info("global model advanced", "run_id", run.ID, "project_id", run.ProjectID, "round", newRound,
		"version", pending, "updates", sum.AggregatedCount, "rejected", len(sum.Rejected))
	s.cancelRound(ctx, run)
	s.applyRetention(ctx, run.ID)
	s.publishRun(ctx, run.ID)
	s.Kick()
	return nil
}

func (s *Service) recordUpdate(ctx context.Context, tx pgx.Tx, run *Run, base *version, u acceptedUpdate, state, why string, resulting *string) error {
	uid := u.UpdateID
	if uid == "" {
		uid = u.AttemptID
	}
	_, err := tx.Exec(ctx, `INSERT INTO training_updates (run_id, update_id, job_id, attempt_id, device_id, base_version, round, update_artifact,
			metadata, state, state_detail, resulting_version, applied_at)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12, CASE WHEN $10='applied' THEN now() END)
		ON CONFLICT (attempt_id) DO UPDATE SET state=EXCLUDED.state, state_detail=EXCLUDED.state_detail,
			resulting_version=EXCLUDED.resulting_version, applied_at=EXCLUDED.applied_at`,
		run.ID, uid, u.JobID, u.AttemptID, u.DeviceID, base.ID, run.Round, nilIfEmpty(u.Artifacts["update_weights"]),
		toJSON(u.Detail), state, why, resulting)
	return err
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func num(v any) float64 {
	if f, ok := v.(float64); ok {
		return f
	}
	return 0
}

// cancelRound cancels jobs of rounds that are already aggregated or abandoned.
func (s *Service) cancelRound(ctx context.Context, run *Run) {
	rows, err := s.d.DB.Query(ctx, `SELECT id FROM jobs WHERE training_run_id=$1 AND status NOT IN ('completed','failed','cancelled')
		AND (model_version IS DISTINCT FROM (SELECT model_version FROM training_runs WHERE id=$1)
		     OR (SELECT status FROM training_runs WHERE id=$1) IN ('stopping','stopped'))`, run.ID)
	if err != nil {
		return
	}
	ids, _ := pgx.CollectRows(rows, pgx.RowTo[string])
	for _, id := range ids {
		_ = s.sched.Cancel(ctx, core.System, run.ProjectID, id)
	}
}

// finalCheckpoint marks the current version as a protected stop checkpoint.
func (s *Service) finalCheckpoint(ctx context.Context, run *Run, trigger string) {
	cur, err := s.runByID(ctx, run.ID)
	if err != nil || cur.ModelVersion == nil {
		return
	}
	_, _ = s.d.DB.Exec(ctx, `INSERT INTO checkpoints (run_id, project_id, model_version, step, round, trigger, manifest, artifact_id, checksum, protected)
		SELECT $1, $2, v.id, $3, $4, $5, v.manifest, v.weights_artifact, COALESCE(v.manifest->>'weights_sha256',''), true
		FROM model_versions v WHERE v.id=$6`, run.ID, run.ProjectID, cur.Step, cur.Round, trigger, *cur.ModelVersion)
}

// applyRetention keeps the newest N round checkpoints plus every protected
// one and the best by held-out loss; older ones are marked deleted (their
// artifacts are retained until a storage GC pass, never the latest).
func (s *Service) applyRetention(ctx context.Context, runID string) {
	keep := s.d.Cfg.CheckpointRetain
	if keep < 2 {
		keep = 2
	}
	_, _ = s.d.DB.Exec(ctx, `UPDATE checkpoints SET protected=true WHERE id = (
		SELECT id FROM checkpoints WHERE run_id=$1 AND state='valid' AND metrics ? 'heldout_loss_after'
		ORDER BY (metrics->>'heldout_loss_after')::float8 ASC LIMIT 1)`, runID)
	_, _ = s.d.DB.Exec(ctx, `UPDATE checkpoints SET state='deleted' WHERE run_id=$1 AND trigger='round' AND NOT protected AND state='valid'
		AND id NOT IN (SELECT id FROM checkpoints WHERE run_id=$1 AND trigger='round' AND state='valid' ORDER BY created_at DESC LIMIT $2)`, runID, keep)
}

func newUUID() string {
	b := secure.RandomBytes(16)
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// initFromBundle registers an existing bundle (weights + manifest) as
// version 0 after checking it was trained against the run's VAE.
func (s *Service) initFromBundle(ctx context.Context, run *Run, pending, vaeSHA string) error {
	wPath, wi, err := s.store.Path(ctx, run.ProjectID, run.Config.InitWeightsArtifact)
	if err != nil {
		s.setStatus(ctx, run.ID, "failed", "init_weights_artifact unavailable")
		return nil
	}
	var man map[string]any
	mPath, err := s.readJSONArtifact(ctx, run.ProjectID, run.Config.InitManifestArtifact, &man)
	if err != nil {
		s.setStatus(ctx, run.ID, "failed", "init_manifest_artifact unreadable")
		return nil
	}
	_ = wPath
	if got, _ := man["vae_sha256"].(string); got != vaeSHA {
		s.setStatus(ctx, run.ID, "failed", "the bundle was trained against a different VAE than vae_artifact")
		return nil
	}
	if got, _ := man["weights_sha256"].(string); got != wi.SHA256 {
		s.setStatus(ctx, run.ID, "failed", "init weights do not match the manifest's weights_sha256")
		return nil
	}
	raw, err := os.ReadFile(mPath)
	if err != nil {
		return err
	}
	paramCount, _ := man["param_count"].(float64)
	arch, _ := man["architecture_version"].(string)
	tokv, _ := man["tokenizer_version"].(string)
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `INSERT INTO model_architectures (id, source_revision, config, tokenizer_version, parameter_count)
			VALUES ($1, $2, $3, $4, $5) ON CONFLICT (id) DO NOTHING`, run.Config.Architecture, arch,
			toJSON(map[string]any{"architecture_version": arch}), tokv, int64(paramCount)); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `INSERT INTO model_versions (id, project_id, run_id, architecture_id, version_number, weights_artifact, manifest_artifact, manifest)
			VALUES ($1,$2,$3,$4,0,$5,$6,$7) ON CONFLICT (id) DO NOTHING`, pending, run.ProjectID, run.ID, run.Config.Architecture,
			run.Config.InitWeightsArtifact, run.Config.InitManifestArtifact, raw); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE training_runs SET model_version=$2, architecture_id=$3, pending_version=NULL, status='running',
			status_detail='initialised from bundle', current_round=0, round_batch=0, round_started_at=NULL, updated_at=now() WHERE id=$1`,
			run.ID, pending, run.Config.Architecture); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE projects SET active_model_version=$2, updated_at=now() WHERE id=$1`, run.ProjectID, pending); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `INSERT INTO checkpoints (run_id, project_id, model_version, step, round, trigger, manifest, artifact_id, checksum, protected)
			VALUES ($1,$2,$3,0,0,'import',$4,$5,$6,true)`, run.ID, run.ProjectID, pending, raw, wi.ID, wi.SHA256)
		return err
	})
	if err != nil {
		return err
	}
	s.d.Log.Info("training run initialised from bundle", "run_id", run.ID, "project_id", run.ProjectID)
	s.publishRun(ctx, run.ID)
	s.Kick()
	return nil
}
