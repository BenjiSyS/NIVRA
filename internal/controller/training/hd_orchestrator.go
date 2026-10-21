package training

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/controller/scheduler"
	"nivra/internal/shared/protocol"
)

// HD distributed training (docs/architecture/HD_TRAINING.md). A run with
// family "hd" trains a DiT (or, optionally, the VAE) on a registered dataset:
// every round hands workers disjoint shards, collects fp16 deltas, aggregates
// them on the controller with ai.hd_aggregate and promotes the result only if
// the held-out gate passes. All advance functions are idempotent: they run
// from the periodic tick and after each finalised job.

// Run families.
const (
	FamilySmall = "small"
	FamilyHD    = "hd"
)

// Stage is the clip geometry a dataset was encoded for / a run trains at.
type Stage struct {
	H      int `json:"h"`
	W      int `json:"w"`
	Frames int `json:"frames"`
	FPS    int `json:"fps"`
}

// Retention keeps the newest KeepLast round checkpoints plus every KeepEvery-th round.
type Retention struct {
	KeepLast  int `json:"keep_last"`
	KeepEvery int `json:"keep_every"`
}

// HDFields are the RunConfig additions for family "hd". They are embedded, so
// the JSON stays flat and every field is omitted for small-model runs.
type HDFields struct {
	Family         string     `json:"family,omitempty"` // "small" (default) | "hd"
	Target         string     `json:"target,omitempty"` // "dit" | "vae"
	HDConfig       string     `json:"hd_config,omitempty"`
	HDConfigSHA    string     `json:"hd_config_sha,omitempty"`
	DatasetID      string     `json:"dataset_id,omitempty"`
	Stage          *Stage     `json:"stage,omitempty"`
	VAEVersionID   string     `json:"vae_version_id,omitempty"`
	TextEncoderID  string     `json:"text_encoder_id,omitempty"`
	ShardsPerJob   int        `json:"shards_per_job,omitempty"`
	DeltaDtype     string     `json:"delta_dtype,omitempty"`
	ReqVRAMMB      int        `json:"req_vram_mb,omitempty"`
	EvalGate       string     `json:"eval_gate,omitempty"` // pi | worker | off
	EvalClips      int        `json:"eval_clips,omitempty"`
	BoundC         float64    `json:"bound_c,omitempty"`
	RedundancyFrac float64    `json:"redundancy_frac,omitempty"`
	Retention      *Retention `json:"retention,omitempty"`
	// InitVersionID is the imported model version (nivra model import) that
	// becomes version 0 of the run.
	InitVersionID string `json:"init_version_id,omitempty"`
}

// hdIsLarge reports configs that need a 24 GB-class GPU (Medium/Large DiT).
func hdIsLarge(cfgName string) bool {
	n := strings.ToLower(cfgName)
	return strings.Contains(n, "medium") || strings.Contains(n, "large")
}

// hdDefaults applies the contract's HD defaults (called before the generic ones).
func (c *RunConfig) hdDefaults() {
	large := hdIsLarge(c.HDConfig)
	if c.Target == "" {
		c.Target = "dit"
	}
	if c.Architecture == "" && c.HDConfig != "" {
		c.Architecture = c.HDConfig
	}
	if c.Rounds == 0 {
		c.Rounds = 20
	}
	if c.WorkersPerRound == 0 {
		c.WorkersPerRound = 4
	}
	if c.MinUpdates == 0 {
		c.MinUpdates = 2
		if c.WorkersPerRound < 2 {
			c.MinUpdates = c.WorkersPerRound
		}
	}
	if c.RoundTimeoutSec == 0 {
		c.RoundTimeoutSec = 7200
	}
	if c.Inner.InnerSteps == 0 {
		c.Inner.InnerSteps = 500
	}
	if c.Inner.BatchSize == 0 {
		c.Inner.BatchSize = 4
	}
	if c.Inner.GradAccum == 0 {
		c.Inner.GradAccum = 2
	}
	if c.Inner.LR == 0 {
		c.Inner.LR = 1e-4
	}
	if c.Inner.WeightDecay == 0 {
		c.Inner.WeightDecay = 0.01
	}
	if c.Inner.WarmupSteps == 0 {
		c.Inner.WarmupSteps = 20
	}
	if c.Inner.CFGDropout == 0 && c.Target == "dit" {
		c.Inner.CFGDropout = 0.1
	}
	if c.ShardsPerJob == 0 {
		c.ShardsPerJob = 2
	}
	if c.DeltaDtype == "" {
		c.DeltaDtype = "float16"
	}
	if c.BoundC == 0 {
		c.BoundC = 3
	}
	if c.LeaseSeconds == 0 {
		c.LeaseSeconds = 900
	}
	if c.ReqVRAMMB == 0 {
		c.ReqVRAMMB = 5500
		if large {
			c.ReqVRAMMB = 22000
		}
	}
	if c.ReqRAMMB == 0 {
		c.ReqRAMMB = 6144
		if large {
			c.ReqRAMMB = 16384
		}
	}
	if c.EvalGate == "" {
		c.EvalGate = "pi"
		if large {
			c.EvalGate = "worker"
		}
	}
	if c.EvalClips == 0 {
		c.EvalClips = 8
	}
	if c.Retention == nil {
		c.Retention = &Retention{KeepLast: 5, KeepEvery: 10}
	}
}

// validateHD bounds the HD fields (the generic checks run afterwards).
func (c RunConfig) validateHD() error {
	bad := func(f string, a ...any) error { return fmt.Errorf("%w: %s", ErrBadConfig, fmt.Sprintf(f, a...)) }
	switch {
	case c.Target != "dit" && c.Target != "vae":
		return bad("target must be dit or vae")
	case !validName(c.HDConfig):
		return bad("hd_config must be a config name")
	case !uuidRe.MatchString(c.DatasetID):
		return bad("dataset_id must be a dataset id")
	case !uuidRe.MatchString(c.InitVersionID):
		return bad("init_version_id (an imported model version) is required")
	case c.ShardsPerJob < 1 || c.ShardsPerJob > 64:
		return bad("shards_per_job 1..64")
	case c.DeltaDtype != "float16":
		return bad("delta_dtype must be float16")
	case c.EvalGate != "pi" && c.EvalGate != "worker" && c.EvalGate != "off":
		return bad("eval_gate must be pi, worker or off")
	case c.EvalClips < 1 || c.EvalClips > 256:
		return bad("eval_clips 1..256")
	case c.BoundC <= 0 || c.BoundC > 100:
		return bad("bound_c (0, 100]")
	case c.RedundancyFrac != 0:
		return bad("redundancy_frac > 0 (duplicate-work cosine checks for untrusted devices) is not implemented; use trusted devices")
	case c.ReqVRAMMB < 1 || c.ReqVRAMMB > 1<<20 || c.ReqRAMMB < 1:
		return bad("req_vram_mb and req_ram_mb must be positive")
	case c.Inner.GradAccum < 1 || c.Inner.GradAccum > 64:
		return bad("inner.grad_accum 1..64")
	case c.Inner.WarmupSteps < 0 || c.Inner.WarmupSteps > c.Inner.InnerSteps:
		return bad("inner.warmup_steps 0..inner_steps")
	case c.Inner.CFGDropout < 0 || c.Inner.CFGDropout >= 1:
		return bad("inner.cfg_dropout [0,1)")
	case c.Inner.WeightDecay < 0 || c.Inner.WeightDecay > 1:
		return bad("inner.weight_decay [0,1]")
	case c.Retention != nil && (c.Retention.KeepLast < 1 || c.Retention.KeepLast > 1000 || c.Retention.KeepEvery < 1):
		return bad("retention.keep_last >= 1 and keep_every >= 1")
	}
	if st := c.Stage; st != nil {
		if st.H < 8 || st.H > 4320 || st.W < 8 || st.W > 7680 || st.Frames < 1 || st.Frames > 2000 || st.FPS < 1 || st.FPS > 120 {
			return bad("stage h/w 8..4320/7680, frames 1..2000, fps 1..120")
		}
	}
	return nil
}

// ---- deterministic shard assignment ----

// splitmix64 is the mixer behind the data order (public domain, Vigna).
func splitmix64(state *uint64) uint64 {
	*state += 0x9E3779B97F4A7C15
	z := *state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// hdPermutation is the epoch's shuffle of n shards: Fisher-Yates from the
// end, seeded with DataSeed XOR epoch (docs/architecture/HD_TRAINING.md).
func hdPermutation(n int, dataSeed int64, epoch int64) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	st := uint64(dataSeed) ^ uint64(epoch)
	for i := n - 1; i > 0; i-- {
		j := int(splitmix64(&st) % uint64(i+1))
		p[i], p[j] = p[j], p[i]
	}
	return p
}

// hdSlot is the global job slot: round*W*8 + batch*W + i.
func hdSlot(round, batch, workers, i int) int { return round*workers*8 + batch*workers + i }

// hdAssign returns, for a slot, the indices (into the train shard list) of its
// k consecutive shards: entry j sits at global position slot*k+j; the epoch is
// position/S and the shard is permutation(epoch)[position mod S].
func hdAssign(shards, k, slot int, dataSeed int64) []int {
	if shards <= 0 || k <= 0 {
		return nil
	}
	perms := map[int64][]int{}
	out := make([]int, 0, k)
	for j := 0; j < k; j++ {
		g := int64(slot)*int64(k) + int64(j)
		epoch, pos := g/int64(shards), int(g%int64(shards))
		p, ok := perms[epoch]
		if !ok {
			p = hdPermutation(shards, dataSeed, epoch)
			perms[epoch] = p
		}
		out = append(out, p[pos])
	}
	return out
}

// hdUpdateID is the deterministic update id of a round job (UUID-shaped), so
// re-creating a job after a crash yields identical params.
func hdUpdateID(runID string, round, batch, i int) string {
	h := sha256.Sum256([]byte(fmt.Sprintf("hd-update:%s:%d:%d:%d", runID, round, batch, i)))
	b := h[:16]
	b[6] = (b[6] & 0x0f) | 0x50
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// hdShardRole is the input role of the n-th data shard of a job. Roles must be
// unique (the worker hands inputs to Python as a dict keyed by role), so the
// k shards are data_shard_000, data_shard_001, ... in the order of params.shards.
func hdShardRole(n int) string { return fmt.Sprintf("data_shard_%03d", n) }

// hdValRole is the input role of the n-th held-out shard of an ai.hd_eval job.
func hdValRole(n int) string { return fmt.Sprintf("val_shard_%03d", n) }

// ---- run creation and initialisation ----

// hdVersionRow is the part of a model version the HD paths need.
type hdVersionRow struct {
	ID          string
	Weights     *string
	Train       *string
	Manifest    *string
	State       *string
	VAEArtifact *string
	Raw         []byte
	Man         map[string]any
}

func (s *Service) hdVersionRow(ctx context.Context, projectID, id string) (*hdVersionRow, error) {
	if !uuidRe.MatchString(id) {
		return nil, errNoVersion
	}
	v := &hdVersionRow{ID: id}
	err := s.d.DB.QueryRow(ctx, `SELECT weights_artifact::text, train_weights_artifact::text, manifest_artifact::text, state_artifact::text,
			vae_artifact::text, manifest FROM model_versions WHERE id=$1 AND project_id=$2`, id, projectID).
		Scan(&v.Weights, &v.Train, &v.Manifest, &v.State, &v.VAEArtifact, &v.Raw)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, errNoVersion
	}
	if err != nil {
		return nil, err
	}
	_ = json.Unmarshal(v.Raw, &v.Man)
	return v, nil
}

func (v *hdVersionRow) str(k string) string { s, _ := v.Man[k].(string); return s }

// trainOrMaster is what workers download as the base: the fp16 train copy
// when the version has one, else the fp32 master.
func (v *hdVersionRow) trainOrMaster() *string {
	if v.Train != nil {
		return v.Train
	}
	return v.Weights
}

// resolveVAE finds the VAE weights and manifest artifacts of a VAE version: a
// target=vae HD version (its train copy or master) or an imported full bundle
// (its vae_artifact).
func (s *Service) resolveVAE(ctx context.Context, projectID, versionID string) (weights, manifest string, err error) {
	v, err := s.hdVersionRow(ctx, projectID, versionID)
	if err != nil {
		return "", "", fmt.Errorf("%w: vae_version_id is not a model version of this project", ErrBadConfig)
	}
	if v.str("target") == "vae" {
		if p := v.trainOrMaster(); p != nil && v.Manifest != nil {
			return *p, *v.Manifest, nil
		}
	} else if v.VAEArtifact != nil && v.Manifest != nil {
		return *v.VAEArtifact, *v.Manifest, nil
	}
	return "", "", fmt.Errorf("%w: that model version has no VAE weights", ErrBadConfig)
}

// createHDRun validates an HD run against its dataset and init version and stores it.
func (s *Service) createHDRun(ctx context.Context, actor core.Actor, projectID, name string, cfg RunConfig) (*Run, error) {
	name = strings.TrimSpace(name)
	if name == "" || len(name) > 80 {
		return nil, fmt.Errorf("%w: name must be 1-80 characters", ErrBadConfig)
	}
	ds, err := s.getDataset(ctx, projectID, cfg.DatasetID)
	if err != nil {
		return nil, fmt.Errorf("%w: dataset not found in this project", ErrBadConfig)
	}
	if ds.Status != "ready" {
		return nil, fmt.Errorf("%w: dataset is %s, not ready", ErrBadConfig, ds.Status)
	}
	wantKind := map[string]string{"dit": "latents", "vae": "raw"}[cfg.Target]
	if ds.Kind != wantKind {
		return nil, fmt.Errorf("%w: target %s trains on a %s dataset", ErrBadConfig, cfg.Target, wantKind)
	}
	if cfg.AllowUntrusted && ds.Redistribution != "any" {
		return nil, fmt.Errorf("%w: this dataset may only be sent to your own trusted devices (redistribution=owner_devices)", ErrBadConfig)
	}
	if cfg.Stage == nil {
		cfg.Stage = ds.Encoder.Stage
	} else if ds.Encoder.Stage != nil && *cfg.Stage != *ds.Encoder.Stage {
		return nil, fmt.Errorf("%w: stage differs from the dataset's encoded stage", ErrBadConfig)
	}
	if cfg.Stage == nil {
		return nil, fmt.Errorf("%w: stage {h,w,frames,fps} is required", ErrBadConfig)
	}
	var vaeWeights *string
	if cfg.Target == "dit" {
		if cfg.VAEVersionID == "" {
			cfg.VAEVersionID = ds.Encoder.VAEVersionID
		}
		if cfg.TextEncoderID == "" {
			cfg.TextEncoderID = ds.Encoder.TextEncoderID
		}
		if cfg.VAEVersionID == "" || cfg.VAEVersionID != ds.Encoder.VAEVersionID || cfg.TextEncoderID != ds.Encoder.TextEncoderID {
			return nil, fmt.Errorf("%w: the latents dataset was encoded with a different VAE / text encoder (%s / %s); encode again or change vae_version_id/text_encoder_id",
				ErrBadConfig, ds.Encoder.VAEVersionID, ds.Encoder.TextEncoderID)
		}
		w, _, err := s.resolveVAE(ctx, projectID, cfg.VAEVersionID)
		if err != nil {
			return nil, err
		}
		vaeWeights = &w
	}
	shards, err := s.datasetShards(ctx, s.d.DB, ds.ID)
	if err != nil {
		return nil, err
	}
	train, val := splitShards(shards)
	if len(train) == 0 {
		return nil, fmt.Errorf("%w: the dataset has no train shards", ErrBadConfig)
	}
	if cfg.ShardsPerJob > len(train) {
		return nil, fmt.Errorf("%w: shards_per_job (%d) exceeds the %d train shards", ErrBadConfig, cfg.ShardsPerJob, len(train))
	}
	if cfg.EvalGate != "off" && len(val) == 0 {
		return nil, fmt.Errorf("%w: eval_gate %s needs at least one val shard; use eval_gate=off to train without a held-out gate", ErrBadConfig, cfg.EvalGate)
	}
	init, err := s.hdVersionRow(ctx, projectID, cfg.InitVersionID)
	if err != nil || init.Weights == nil || init.Manifest == nil {
		return nil, fmt.Errorf("%w: init_version_id is not a model version with weights and manifest in this project", ErrBadConfig)
	}
	if f := init.str("family"); f != "" && f != FamilyHD {
		return nil, fmt.Errorf("%w: the init version is a %q model, not an HD model", ErrBadConfig, f)
	}
	if t := init.str("target"); t != "" && t != cfg.Target {
		return nil, fmt.Errorf("%w: the init version trains target %q, not %q", ErrBadConfig, t, cfg.Target)
	}
	manSHA := init.str("hd_config_sha")
	switch {
	case cfg.HDConfigSHA == "" && manSHA == "":
		return nil, fmt.Errorf("%w: hd_config_sha is required (the init manifest carries none)", ErrBadConfig)
	case cfg.HDConfigSHA == "":
		cfg.HDConfigSHA = manSHA
	case manSHA != "" && manSHA != cfg.HDConfigSHA:
		return nil, fmt.Errorf("%w: hd_config_sha does not match the init version's manifest", ErrBadConfig)
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	b, _ := json.Marshal(cfg)
	var by any
	if actor.Type == "user" {
		by = actor.ID
	}
	var id string
	if err := s.d.DB.QueryRow(ctx, `INSERT INTO training_runs (project_id, name, config, status, vae_artifact, dataset_version, created_by)
		VALUES ($1,$2,$3,'created',$4,$5,$6) RETURNING id`, projectID, name, b, vaeWeights, "dataset:"+ds.ID+"/"+ds.Fingerprint, by).Scan(&id); err != nil {
		return nil, err
	}
	s.d.Audit(ctx, nil, actor, "training.create", "run:"+id, "success", map[string]any{"name": name, "family": FamilyHD, "dataset": ds.ID})
	s.Kick()
	return s.GetRun(ctx, projectID, id)
}

// initHDRun creates global version 0 as a copy of the init version.
func (s *Service) initHDRun(ctx context.Context, run *Run) error {
	if run.ModelVersion != nil {
		s.setStatus(ctx, run.ID, "running", "")
		return nil
	}
	c := run.Config
	init, err := s.hdVersionRow(ctx, run.ProjectID, c.InitVersionID)
	if err != nil || init.Weights == nil || init.Manifest == nil {
		s.setStatus(ctx, run.ID, "failed", "the init version is gone or has no weights/manifest")
		return nil
	}
	_, _ = s.d.DB.Exec(ctx, `UPDATE training_runs SET status='initializing', status_detail='creating global version 0', updated_at=now(),
		pending_version=COALESCE(pending_version, gen_random_uuid()) WHERE id=$1`, run.ID)
	var pending string
	if err := s.d.DB.QueryRow(ctx, `SELECT pending_version::text FROM training_runs WHERE id=$1`, run.ID).Scan(&pending); err != nil {
		return err
	}
	params, _ := init.Man["param_count"].(float64)
	arch := c.HDConfig
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `INSERT INTO model_architectures (id, source_revision, config, tokenizer_version, parameter_count)
			VALUES ($1,$2,$3,$4,$5) ON CONFLICT (id) DO NOTHING`, arch, init.str("architecture_version"),
			toJSON(map[string]any{"architecture_version": init.str("architecture_version")}), init.str("tokenizer_version"), int64(params)); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `INSERT INTO model_versions (id, project_id, run_id, architecture_id, version_number, weights_artifact, manifest_artifact,
				state_artifact, train_weights_artifact, vae_artifact, manifest, family, compat_sha)
			VALUES ($1,$2,$3,$4,0,$5,$6,$7,$8,$9,$10,'hd',$11) ON CONFLICT (id) DO NOTHING`,
			pending, run.ProjectID, run.ID, arch, init.Weights, init.Manifest, init.State, init.Train, run.VAEArtifact, init.Raw, c.HDConfigSHA); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE training_runs SET model_version=$2, architecture_id=$3, pending_version=NULL, status='running',
			status_detail='initialised from model version', current_round=0, round_batch=0, round_started_at=NULL, updated_at=now() WHERE id=$1`,
			run.ID, pending, arch); err != nil {
			return err
		}
		if c.Target == "dit" {
			if _, err := tx.Exec(ctx, `UPDATE projects SET active_model_version=$2, updated_at=now() WHERE id=$1 AND active_model_version IS NULL`, run.ProjectID, pending); err != nil {
				return err
			}
		}
		_, err := tx.Exec(ctx, `INSERT INTO checkpoints (run_id, project_id, model_version, step, round, trigger, manifest, artifact_id, checksum, protected)
			SELECT $1,$2,$3,0,0,'import',$4,$5,COALESCE($4::jsonb->>'weights_sha256',''),true`, run.ID, run.ProjectID, pending, init.Raw, init.Weights)
		return err
	})
	if err != nil {
		return err
	}
	s.d.Log.Info("HD training run initialised", "run_id", run.ID, "project_id", run.ProjectID, "target", c.Target)
	s.publishRun(ctx, run.ID)
	s.Kick()
	return nil
}

// ---- the state machine ----

func (s *Service) stepHD(ctx context.Context, run *Run) error {
	switch run.Status {
	case "created", "initializing":
		return s.initHDRun(ctx, run)
	case "running":
		if run.Round >= run.Config.Rounds {
			s.finalCheckpoint(ctx, run, "stop")
			s.setStatus(ctx, run.ID, "completed", fmt.Sprintf("completed %d rounds", run.Round))
			return nil
		}
		cand, err := s.evaluatingCandidate(ctx, run)
		if err != nil {
			return err
		}
		if cand != nil {
			_, err := s.advanceCandidate(ctx, run, cand)
			return err
		}
		if err := s.ensureHDRoundJobs(ctx, run); err != nil {
			return err
		}
		return s.maybeAggregateHD(ctx, run, false)
	case "stopping":
		cand, err := s.evaluatingCandidate(ctx, run)
		if err != nil {
			return err
		}
		if cand != nil {
			done, err := s.advanceCandidate(ctx, run, cand)
			if err != nil || !done {
				return err
			}
			s.finishHDStop(ctx, run)
			return nil
		}
		if err := s.maybeAggregateHD(ctx, run, true); err != nil {
			return err
		}
		if c2, err := s.evaluatingCandidate(ctx, run); err != nil || c2 != nil {
			return err // wait for the held-out gate before stopping
		}
		s.finishHDStop(ctx, run)
	}
	return nil
}

func (s *Service) finishHDStop(ctx context.Context, run *Run) {
	s.cancelRound(ctx, run)
	s.finalCheckpoint(ctx, run, "stop")
	s.setStatus(ctx, run.ID, "stopped", fmt.Sprintf("stopped after round %d", run.Round))
}

// loadRoundHD is loadRound without updates the aggregator already invalidated
// (otherwise a rejected round would re-aggregate the same updates forever).
func (s *Service) loadRoundHD(ctx context.Context, run *Run) (*roundJobs, error) {
	rj, err := s.loadRound(ctx, run)
	if err != nil {
		return nil, err
	}
	keep := rj.accepted[:0]
	for _, u := range rj.accepted {
		var dead bool
		if err := s.d.DB.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM training_updates WHERE attempt_id=$1 AND state IN ('invalid','stale','applied'))`,
			u.AttemptID).Scan(&dead); err != nil {
			return nil, err
		}
		if !dead {
			keep = append(keep, u)
		}
	}
	rj.accepted = keep
	return rj, nil
}

type hdTrainParams struct {
	UpdateID      string     `json:"update_id"`
	RunID         string     `json:"run_id"`
	Round         int        `json:"round"`
	RoundBatch    int        `json:"round_batch"`
	BaseVersionID string     `json:"base_version_id"`
	Target        string     `json:"target"`
	HDConfig      string     `json:"hd_config"`
	HDConfigSHA   string     `json:"hd_config_sha"`
	Stage         Stage      `json:"stage"`
	InnerSteps    int        `json:"inner_steps"`
	BatchSize     int        `json:"batch_size"`
	GradAccum     int        `json:"grad_accum"`
	LR            float64    `json:"lr"`
	WarmupSteps   int        `json:"warmup_steps"`
	WeightDecay   float64    `json:"weight_decay"`
	GradClip      float64    `json:"grad_clip"`
	CFGDropout    float64    `json:"cfg_dropout"`
	Seed          int        `json:"seed"`
	Shards        []shardRef `json:"shards"`
	DeltaDtype    string     `json:"delta_dtype"`
}

type shardRef struct {
	ArtifactID string `json:"artifact_id"`
	SHA256     string `json:"sha256"`
}

// ensureHDRoundJobs creates the round's ai.hd_train jobs (idempotently) and
// opens a new batch when every job of the batch ended without enough updates.
func (s *Service) ensureHDRoundJobs(ctx context.Context, run *Run) error {
	rj, err := s.loadRoundHD(ctx, run)
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
			s.d.Log.Warn("HD round batch ended without enough valid updates; retrying", "run_id", run.ID, "round", run.Round)
			s.Kick()
			return err
		}
		return nil
	}
	c := run.Config
	need := c.WorkersPerRound - len(rj.accepted)
	if need <= 0 {
		return nil
	}
	ds, err := s.getDataset(ctx, run.ProjectID, c.DatasetID)
	if err != nil {
		return err
	}
	all, err := s.datasetShards(ctx, s.d.DB, ds.ID)
	if err != nil {
		return err
	}
	train, _ := splitShards(all)
	if len(train) == 0 || c.ShardsPerJob > len(train) {
		s.setStatus(ctx, run.ID, "paused", "the dataset no longer has enough train shards")
		return nil
	}
	base, err := s.hdVersionRow(ctx, run.ProjectID, *run.ModelVersion)
	if err != nil {
		return err
	}
	bv, err := s.version(ctx, *run.ModelVersion)
	if err != nil {
		return err
	}
	if base.trainOrMaster() == nil || base.Manifest == nil {
		return errors.New("base version has no weights/manifest")
	}
	baseVersionID := bv.VersionID
	if baseVersionID == "" {
		baseVersionID = bv.ID
	}
	trusted := !(c.AllowUntrusted && ds.Redistribution == "any")
	stage := Stage{}
	if c.Stage != nil {
		stage = *c.Stage
	}
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		for i := 0; i < need; i++ {
			slot := hdSlot(run.Round, run.RoundBatch, c.WorkersPerRound, i)
			picks := hdAssign(len(train), c.ShardsPerJob, slot, int64(c.DataSeed))
			seen := map[int]bool{}
			var refs []shardRef
			inputs := []protocol.ArtifactRef{{ID: *base.trainOrMaster(), Role: "base_train_weights"}, {ID: *base.Manifest, Role: "base_manifest"}}
			for _, p := range picks {
				if seen[p] {
					continue // a job straddling an epoch boundary can draw a shard twice; load it once
				}
				seen[p] = true
				sh := train[p]
				inputs = append(inputs, protocol.ArtifactRef{ID: sh.ArtifactID, Role: hdShardRole(len(refs))})
				refs = append(refs, shardRef{ArtifactID: sh.ArtifactID, SHA256: sh.SHA256})
			}
			params := hdTrainParams{UpdateID: hdUpdateID(run.ID, run.Round, run.RoundBatch, i), RunID: run.ID, Round: run.Round,
				RoundBatch: run.RoundBatch, BaseVersionID: baseVersionID, Target: c.Target, HDConfig: c.HDConfig, HDConfigSHA: c.HDConfigSHA,
				Stage: stage, InnerSteps: c.Inner.InnerSteps, BatchSize: c.Inner.BatchSize, GradAccum: c.Inner.GradAccum, LR: c.Inner.LR,
				WarmupSteps: c.Inner.WarmupSteps, WeightDecay: c.Inner.WeightDecay, GradClip: c.Inner.GradClip, CFGDropout: c.Inner.CFGDropout,
				Seed: c.Seed*1000003 + slot, Shards: refs, DeltaDtype: c.DeltaDtype}
			pb, _ := json.Marshal(params)
			mv, rid := *run.ModelVersion, run.ID
			if _, err := s.sched.Create(ctx, tx, scheduler.NewJob{ProjectID: run.ProjectID, JobType: "ai.hd_train", Params: pb, Inputs: inputs,
				ReqRAMMB: c.ReqRAMMB, ReqVRAMMB: c.ReqVRAMMB, ReqGPU: true, Priority: 10, MaxAttempts: 3, LeaseSeconds: c.LeaseSeconds,
				ReqTrusted: trusted, ModelVersion: &mv, TrainingRunID: &rid,
				IdempotencyKey: fmt.Sprintf("%sb%d:w%d", roundKeyPrefix(run.ID, run.Round), run.RoundBatch, i)}); err != nil {
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
	s.d.Log.Info("HD training round jobs created", "run_id", run.ID, "project_id", run.ProjectID, "round", run.Round, "batch", run.RoundBatch, "jobs", need)
	s.publishRun(ctx, run.ID)
	return nil
}

func (s *Service) maybeAggregateHD(ctx context.Context, run *Run, force bool) error {
	if run.ModelVersion == nil {
		return nil
	}
	rj, err := s.loadRoundHD(ctx, run)
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
	return s.aggregateHD(ctx, run, rj.accepted)
}

// hdAggSummary is the ai.hd_aggregate result (the fields the controller uses).
type hdAggSummary struct {
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

func (a hdAggSummary) rejectedMap() map[string]string {
	m := map[string]string{}
	for _, r := range a.Rejected {
		m[r.UpdateID] = r.Reason
	}
	return m
}

func (a hdAggSummary) appliedMap() map[string]bool {
	m := map[string]bool{}
	for _, id := range a.Applied {
		m[id] = true
	}
	return m
}

func (s *Service) aggregateHD(ctx context.Context, run *Run, ups []acceptedUpdate) error {
	c := run.Config
	if _, err := s.d.DB.Exec(ctx, `UPDATE training_runs SET pending_version=COALESCE(pending_version, gen_random_uuid()), status_detail=$2 WHERE id=$1`,
		run.ID, fmt.Sprintf("round %d: aggregating %d update(s)", run.Round, len(ups))); err != nil {
		return err
	}
	var pending string
	if err := s.d.DB.QueryRow(ctx, `SELECT pending_version::text FROM training_runs WHERE id=$1`, run.ID).Scan(&pending); err != nil {
		return err
	}
	s.publishRun(ctx, run.ID)
	base, err := s.hdVersionRow(ctx, run.ProjectID, *run.ModelVersion)
	if err != nil || base.Weights == nil || base.Manifest == nil {
		return fmt.Errorf("base version unavailable: %v", err)
	}
	path := func(id string) (string, error) {
		p, _, err := s.store.Path(ctx, run.ProjectID, id)
		return p, err
	}
	bw, err := path(*base.Weights)
	if err != nil {
		return err
	}
	bm, err := path(*base.Manifest)
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
		list = append(list, map[string]string{"path": wp, "manifest_path": mp, "update_id": u.UpdateID})
	}
	if len(list) == 0 {
		return errors.New("no readable update artifacts")
	}
	inputs["updates"] = list
	if c.EvalGate == "pi" {
		_, val, err := s.datasetSplits(ctx, run.ProjectID, c.DatasetID)
		if err != nil {
			return err
		}
		var lat []string
		for _, sh := range evalShards(val, c.EvalClips) {
			if p, err := path(sh.ArtifactID); err == nil {
				lat = append(lat, p)
			}
		}
		if len(lat) == 0 {
			return errors.New("eval_gate=pi but no readable val shards")
		}
		inputs["eval_latents"] = lat
	}
	res, _, cleanup, err := s.runPython(ctx, "ai.hd_aggregate", run.ID, map[string]any{
		"outer_lr": c.OuterLR, "momentum": c.Momentum, "nesterov": true, "version_id": pending, "bound_c": c.BoundC,
		"inner_steps": c.Inner.InnerSteps, "lr": c.Inner.LR, "eval_gate": c.EvalGate, "eval_triples_seed": hdTriplesSeed(c),
	}, inputs)
	if err != nil {
		return err
	}
	defer cleanup()
	var sum hdAggSummary
	if err := json.Unmarshal(res.Summary, &sum); err != nil {
		return fmt.Errorf("hd_aggregate summary: %w", err)
	}
	if sum.NoOp || sum.NewVersionID == nil {
		return s.rejectRound(ctx, run, ups, "aggregation rejected every update", sum.rejectedMap(), nil)
	}
	// pi gate: the Python side measured held-out loss of the aggregate.
	if c.EvalGate == "pi" && sum.HeldoutBefore != nil && sum.HeldoutAfter != nil && c.MaxHeldoutIncrease > 0 &&
		*sum.HeldoutAfter > *sum.HeldoutBefore*(1+c.MaxHeldoutIncrease) {
		why := fmt.Sprintf("aggregated version rejected: held-out loss %.4f -> %.4f", *sum.HeldoutBefore, *sum.HeldoutAfter)
		return s.rejectRound(ctx, run, ups, why, nil, nil)
	}
	w, ok1 := artifactByKind(res, "model_weights")
	t, ok2 := artifactByKind(res, "model_train_weights")
	m, ok3 := artifactByKind(res, "model_manifest")
	if !ok3 {
		m, ok3 = artifactByKind(res, "model_version")
	}
	if !ok1 || !ok2 || !ok3 {
		return errors.New("ai.hd_aggregate must produce model_weights, model_train_weights and model_manifest artifacts")
	}
	wi, err := s.store.Import(ctx, run.ProjectID, "model_weights", w.Path, "application/octet-stream")
	if err != nil {
		return err
	}
	ti, err := s.store.Import(ctx, run.ProjectID, "model_train_weights", t.Path, "application/octet-stream")
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
	out := &hdOutcome{Pending: pending, Weights: wi.ID, Train: ti.ID, Manifest: mi.ID, State: stateID, Man: man, WeightsSHA: wi.SHA256,
		Sum: sum, Applied: sum.appliedMap(), RejectedWhy: sum.rejectedMap(), Started: time.Now()}
	if c.EvalGate == "worker" {
		return s.openCandidate(ctx, run, base, out)
	}
	return s.promoteHD(ctx, run, base.ID, out, ups, nil)
}

// hdTriplesSeed fixes the held-out (latent, t, noise) triples for the whole run
// so the loss is comparable across rounds.
func hdTriplesSeed(c RunConfig) int { return c.Seed*1000003 + 7 }

// hdOutcome is an aggregated, imported version waiting to be promoted.
type hdOutcome struct {
	Pending     string
	Weights     string
	Train       string
	Manifest    string
	State       *string
	Man         []byte
	WeightsSHA  string
	Sum         hdAggSummary
	Applied     map[string]bool
	RejectedWhy map[string]string
	Started     time.Time
}

// rejectRound records every update of the attempt as invalid and opens a new batch.
func (s *Service) rejectRound(ctx context.Context, run *Run, ups []acceptedUpdate, why string, perUpdate map[string]string, candidate *string) error {
	base, err := s.version(ctx, *run.ModelVersion)
	if err != nil {
		return err
	}
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		for _, u := range ups {
			reason := why
			if r := perUpdate[u.UpdateID]; r != "" {
				reason = r
			}
			if err := s.recordUpdate(ctx, tx, run, base, u, "invalid", reason, nil); err != nil {
				return err
			}
		}
		if candidate != nil {
			if _, err := tx.Exec(ctx, `UPDATE hd_candidates SET state='rejected', detail=$2 WHERE version_id=$1 AND state='evaluating'`, *candidate, truncate(why, 500)); err != nil {
				return err
			}
		}
		_, err := tx.Exec(ctx, `UPDATE training_runs SET round_batch=round_batch+1, pending_version=NULL, status_detail=$2, updated_at=now() WHERE id=$1`,
			run.ID, truncate(why, 500))
		return err
	})
	if err != nil {
		return err
	}
	s.d.Log.Warn("HD round rejected; retrying with new shards", "run_id", run.ID, "round", run.Round, "reason", why)
	s.publishRun(ctx, run.ID)
	s.Kick()
	return nil
}

// promoteHD makes the aggregated version the run's current one. It is safe to
// call twice: the run row is locked and nothing happens once it has moved on.
func (s *Service) promoteHD(ctx context.Context, run *Run, baseID string, o *hdOutcome, ups []acceptedUpdate, evalLoss *float64) error {
	c := run.Config
	var lossFirst, lossLast, samples float64
	devices := map[string]bool{}
	for _, u := range ups {
		lossFirst += num(u.Detail["loss_first"])
		lossLast += num(u.Detail["loss_last"])
		samples += num(u.Detail["samples_seen"])
		devices[u.DeviceID] = true
	}
	nu := float64(len(ups))
	if nu == 0 {
		nu = 1
	}
	newRound := run.Round + 1
	newStep := run.Step + int64(c.Inner.InnerSteps)
	promoted := false
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var cur *string
		var curRound int
		if err := tx.QueryRow(ctx, `SELECT model_version::text, current_round FROM training_runs WHERE id=$1 FOR UPDATE`, run.ID).Scan(&cur, &curRound); err != nil {
			return err
		}
		if cur == nil || *cur != baseID || curRound != run.Round {
			return nil // already promoted (or the run was rolled back): nothing to do
		}
		var number int
		if err := tx.QueryRow(ctx, `SELECT version_number FROM model_versions WHERE id=$1`, baseID).Scan(&number); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `INSERT INTO model_versions (id, project_id, run_id, architecture_id, version_number, parent_version, weights_artifact,
				manifest_artifact, state_artifact, train_weights_artifact, vae_artifact, manifest, family, compat_sha)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,'hd',$13) ON CONFLICT (id) DO NOTHING`,
			o.Pending, run.ProjectID, run.ID, c.HDConfig, number+1, baseID, o.Weights, o.Manifest, o.State, o.Train, run.VAEArtifact, o.Man, c.HDConfigSHA); err != nil {
			return err
		}
		base, err := s.versionTx(ctx, tx, baseID)
		if err != nil {
			return err
		}
		for _, u := range ups {
			state, why := "applied", ""
			if !o.Applied[u.UpdateID] {
				state, why = "invalid", o.RejectedWhy[u.UpdateID]
				if why == "" {
					why = "not applied by aggregator"
				}
			}
			var res *string
			if state == "applied" {
				res = &o.Pending
			}
			if err := s.recordUpdate(ctx, tx, run, base, u, state, why, res); err != nil {
				return err
			}
		}
		if _, err := tx.Exec(ctx, `UPDATE training_runs SET model_version=$2, current_round=$3, current_step=$4, round_batch=0,
			round_started_at=NULL, pending_version=NULL, status_detail=$5, updated_at=now() WHERE id=$1`,
			run.ID, o.Pending, newRound, newStep, fmt.Sprintf("round %d aggregated (%d update(s))", run.Round, o.Sum.AggregatedCount)); err != nil {
			return err
		}
		if c.Target == "dit" {
			if _, err := tx.Exec(ctx, `UPDATE projects SET active_model_version=$2, updated_at=now() WHERE id=$1 AND (active_model_version IS NULL OR active_model_version=$3)`,
				run.ProjectID, o.Pending, baseID); err != nil {
				return err
			}
		}
		metrics := map[string]any{"delta_l2": o.Sum.DeltaL2, "aggregated": o.Sum.AggregatedCount, "method": o.Sum.Method}
		if o.Sum.HeldoutAfter != nil {
			metrics["heldout_loss_after"] = *o.Sum.HeldoutAfter
		}
		if evalLoss != nil {
			metrics["heldout_loss_after"] = *evalLoss
		}
		if _, err := tx.Exec(ctx, `INSERT INTO checkpoints (run_id, project_id, model_version, step, round, trigger, manifest, artifact_id, checksum, metrics)
			VALUES ($1,$2,$3,$4,$5,'round',$6,$7,$8,$9)`, run.ID, run.ProjectID, o.Pending, newStep, newRound, o.Man, o.Weights, o.WeightsSHA, toJSON(metrics)); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE hd_candidates SET state='accepted', eval_loss=COALESCE($2, eval_loss) WHERE version_id=$1`, o.Pending, evalLoss); err != nil {
			return err
		}
		ins := func(name string, v float64) error {
			_, err := tx.Exec(ctx, `INSERT INTO training_metrics (run_id, round, step, name, value) VALUES ($1,$2,$3,$4,$5)`, run.ID, newRound, newStep, name, v)
			return err
		}
		vals := map[string]float64{"accepted_updates": float64(o.Sum.AggregatedCount), "rejected_updates": float64(len(o.Sum.Rejected)),
			"delta_l2": o.Sum.DeltaL2, "update_loss_first_mean": lossFirst / nu, "update_loss_last_mean": lossLast / nu,
			"round_seconds": time.Since(o.Started).Seconds(), "active_workers": float64(len(devices)), "outer_lr": c.OuterLR, "samples_seen": samples}
		if run.RoundStarted != nil {
			vals["round_seconds"] = time.Since(*run.RoundStarted).Seconds()
		}
		if o.Sum.HeldoutBefore != nil {
			vals["heldout_loss_before"] = *o.Sum.HeldoutBefore
		}
		if o.Sum.HeldoutAfter != nil {
			vals["heldout_loss_after"] = *o.Sum.HeldoutAfter
		}
		if evalLoss != nil {
			vals["heldout_loss_after"] = *evalLoss
		}
		for k, v := range vals {
			if err := ins(k, v); err != nil {
				return err
			}
		}
		promoted = true
		return nil
	})
	if err != nil {
		return err
	}
	if !promoted {
		return nil
	}
	s.d.Log.Info("HD global model advanced", "run_id", run.ID, "project_id", run.ProjectID, "round", newRound, "version", o.Pending,
		"updates", o.Sum.AggregatedCount, "rejected", len(o.Sum.Rejected))
	s.cancelRound(ctx, run)
	s.applyHDRetention(ctx, run.ID, c.Retention)
	s.publishRun(ctx, run.ID)
	s.Kick()
	return nil
}

// versionTx loads the version columns recordUpdate needs, inside a transaction.
func (s *Service) versionTx(ctx context.Context, tx pgx.Tx, id string) (*version, error) {
	v := &version{ID: id}
	err := tx.QueryRow(ctx, `SELECT version_number, weights_artifact::text, manifest_artifact::text FROM model_versions WHERE id=$1`, id).
		Scan(&v.Number, &v.Weights, &v.Manifest)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, errNoVersion
	}
	return v, err
}

// applyHDRetention keeps the newest KeepLast round checkpoints, every
// KeepEvery-th round, protected ones and the best by held-out loss; the rest
// are marked deleted (artifacts stay until storage GC, never the latest).
func (s *Service) applyHDRetention(ctx context.Context, runID string, r *Retention) {
	keepLast, keepEvery := 5, 10
	if r != nil {
		keepLast, keepEvery = r.KeepLast, r.KeepEvery
	}
	_, _ = s.d.DB.Exec(ctx, `UPDATE checkpoints SET protected=true WHERE id = (
		SELECT id FROM checkpoints WHERE run_id=$1 AND state='valid' AND metrics ? 'heldout_loss_after'
		ORDER BY (metrics->>'heldout_loss_after')::float8 ASC LIMIT 1)`, runID)
	_, _ = s.d.DB.Exec(ctx, `UPDATE checkpoints SET state='deleted' WHERE run_id=$1 AND trigger='round' AND NOT protected AND state='valid'
		AND round % $3 <> 0
		AND id NOT IN (SELECT id FROM checkpoints WHERE run_id=$1 AND trigger='round' AND state='valid' ORDER BY created_at DESC LIMIT $2)`,
		runID, keepLast, keepEvery)
}

// ---- worker-evaluated held-out gate ----

type hdCandidate struct {
	VersionID string
	Round     int
	Weights   string
	Train     string
	Manifest  string
	State     *string
	Man       []byte
	Out       *hdOutcome
}

func (s *Service) openCandidate(ctx context.Context, run *Run, base *hdVersionRow, o *hdOutcome) error {
	sum, _ := json.Marshal(o.Sum)
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, `INSERT INTO hd_candidates (version_id, run_id, base_version, round, weights_artifact, train_artifact, manifest_artifact,
				state_artifact, manifest, summary) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) ON CONFLICT (version_id) DO NOTHING`,
			o.Pending, run.ID, base.ID, run.Round, o.Weights, o.Train, o.Manifest, o.State, o.Man, sum)
		if err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `UPDATE training_runs SET status_detail=$2, updated_at=now() WHERE id=$1`, run.ID,
			fmt.Sprintf("round %d: aggregated; waiting for the held-out evaluation on a trusted worker", run.Round))
		return err
	})
	if err != nil {
		return err
	}
	s.publishRun(ctx, run.ID)
	s.Kick()
	return nil
}

func (s *Service) evaluatingCandidate(ctx context.Context, run *Run) (*hdCandidate, error) {
	if run.ModelVersion == nil {
		return nil, nil
	}
	cand := &hdCandidate{}
	var sum []byte
	var w, t, m *string
	err := s.d.DB.QueryRow(ctx, `SELECT version_id::text, round, weights_artifact::text, train_artifact::text, manifest_artifact::text, state_artifact::text,
			manifest, summary FROM hd_candidates WHERE run_id=$1 AND base_version=$2 AND state='evaluating' ORDER BY created_at DESC LIMIT 1`,
		run.ID, *run.ModelVersion).Scan(&cand.VersionID, &cand.Round, &w, &t, &m, &cand.State, &cand.Man, &sum)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if w == nil || t == nil || m == nil {
		return nil, errors.New("candidate artifacts missing")
	}
	cand.Weights, cand.Train, cand.Manifest = *w, *t, *m
	o := &hdOutcome{Pending: cand.VersionID, Weights: *w, Train: *t, Manifest: *m, State: cand.State, Man: cand.Man, Started: time.Now()}
	if err := json.Unmarshal(sum, &o.Sum); err != nil {
		return nil, err
	}
	o.Applied, o.RejectedWhy = o.Sum.appliedMap(), o.Sum.rejectedMap()
	var wsha string
	_ = s.d.DB.QueryRow(ctx, `SELECT sha256 FROM artifacts WHERE id=$1`, *w).Scan(&wsha)
	o.WeightsSHA = wsha
	cand.Out = o
	return cand, nil
}

type hdEvalState struct {
	jobID   string
	status  string
	failure string
	loss    *float64
}

// ensureEvalJob creates (idempotently) the ai.hd_eval job of a version and
// reports its state. One job per run+version: the key is hdeval:{run}:{version}.
func (s *Service) ensureEvalJob(ctx context.Context, run *Run, versionID string, weights, manifest *string, val []datasetShard) (*hdEvalState, error) {
	c := run.Config
	key := "hdeval:" + run.ID + ":" + versionID
	inputs := []protocol.ArtifactRef{{ID: *weights, Role: "weights"}, {ID: *manifest, Role: "manifest"}}
	for i, sh := range evalShards(val, c.EvalClips) {
		inputs = append(inputs, protocol.ArtifactRef{ID: sh.ArtifactID, Role: hdValRole(i)})
	}
	params, _ := json.Marshal(map[string]any{"version_id": versionID, "triples_seed": hdTriplesSeed(c), "n": c.EvalClips, "timestep_buckets": 4})
	mv, rid := *run.ModelVersion, run.ID
	id, err := s.sched.Create(ctx, nil, scheduler.NewJob{ProjectID: run.ProjectID, JobType: "ai.hd_eval", Params: params, Inputs: inputs,
		ReqRAMMB: c.ReqRAMMB, ReqVRAMMB: c.ReqVRAMMB, ReqGPU: true, Priority: 12, MaxAttempts: 3, LeaseSeconds: c.LeaseSeconds,
		ReqTrusted: true, ModelVersion: &mv, TrainingRunID: &rid, IdempotencyKey: key})
	if err != nil {
		return nil, err
	}
	st := &hdEvalState{jobID: id}
	if err := s.d.DB.QueryRow(ctx, `SELECT status, failure_reason FROM jobs WHERE id=$1`, id).Scan(&st.status, &st.failure); err != nil {
		return nil, err
	}
	if st.status == scheduler.JobCompleted {
		var loss *float64
		if err := s.d.DB.QueryRow(ctx, `SELECT (validation_detail->>'loss')::float8 FROM job_results WHERE job_id=$1 AND validation_status='accepted'`, id).Scan(&loss); err != nil {
			return nil, err
		}
		st.loss = loss
	}
	return st, nil
}

// advanceCandidate drives the worker gate: evaluate the candidate (and the
// base, unless its loss is already known), then promote or reject. done is
// true once the candidate is resolved either way.
func (s *Service) advanceCandidate(ctx context.Context, run *Run, cand *hdCandidate) (done bool, err error) {
	c := run.Config
	_, val, err := s.datasetSplits(ctx, run.ProjectID, c.DatasetID)
	if err != nil {
		return false, err
	}
	base, err := s.hdVersionRow(ctx, run.ProjectID, *run.ModelVersion)
	if err != nil || base.trainOrMaster() == nil || base.Manifest == nil {
		return false, fmt.Errorf("base version unavailable: %v", err)
	}
	rj, err := s.loadRoundHD(ctx, run)
	if err != nil {
		return false, err
	}
	fail := func(why string) (bool, error) {
		return true, s.rejectRound(ctx, run, rj.accepted, why, nil, &cand.VersionID)
	}
	if len(val) == 0 {
		return fail("the dataset has no val shards for the held-out evaluation")
	}
	ce, err := s.ensureEvalJob(ctx, run, cand.VersionID, &cand.Train, &cand.Manifest, val)
	if err != nil {
		return false, err
	}
	var baseLoss *float64
	_ = s.d.DB.QueryRow(ctx, `SELECT eval_loss FROM hd_candidates WHERE run_id=$1 AND version_id=$2 AND eval_loss IS NOT NULL`, run.ID, run.ModelVersion).Scan(&baseLoss)
	var be *hdEvalState
	if baseLoss == nil {
		be, err = s.ensureEvalJob(ctx, run, base.ID, base.trainOrMaster(), base.Manifest, val)
		if err != nil {
			return false, err
		}
		if be.status == scheduler.JobFailed || be.status == scheduler.JobCancelled {
			return fail("held-out evaluation of the base version failed: " + be.failure)
		}
		baseLoss = be.loss
	}
	if ce.status == scheduler.JobFailed || ce.status == scheduler.JobCancelled {
		return fail("held-out evaluation of the new version failed: " + ce.failure)
	}
	if ce.loss == nil || baseLoss == nil {
		_, _ = s.d.DB.Exec(ctx, `UPDATE training_runs SET status_detail=$2, updated_at=now() WHERE id=$1`, run.ID,
			fmt.Sprintf("round %d: waiting for the held-out evaluation on a trusted worker", cand.Round))
		return false, nil
	}
	if c.MaxHeldoutIncrease > 0 && *ce.loss > *baseLoss*(1+c.MaxHeldoutIncrease) {
		return fail(fmt.Sprintf("aggregated version rejected: held-out loss %.4f -> %.4f", *baseLoss, *ce.loss))
	}
	cand.Out.Sum.HeldoutBefore, cand.Out.Sum.HeldoutAfter = baseLoss, ce.loss
	return true, s.promoteHD(ctx, run, base.ID, cand.Out, rj.accepted, ce.loss)
}
