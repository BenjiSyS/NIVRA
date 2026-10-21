package training

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"nivra/internal/controller/scheduler"
)

// readJSONArtifact loads a small JSON artifact.
// maxManifestBytes bounds JSON artifacts read by the controller (and passed
// to Python), so a padded manifest cannot exhaust the Pi's memory.
const maxManifestBytes = 1 << 20

// errJSONTooLarge marks an oversized JSON artifact (a worker fault, not an
// infrastructure error, so the result is rejected instead of retried).
var errJSONTooLarge = errors.New("JSON artifact exceeds the size limit")

func (s *Service) readJSONArtifact(ctx context.Context, projectID, id string, v any) (string, error) {
	p, info, err := s.store.Path(ctx, projectID, id)
	if err != nil {
		return "", err
	}
	if info.Size > maxManifestBytes {
		return p, errJSONTooLarge
	}
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, maxManifestBytes+1))
	if err != nil {
		return "", err
	}
	if len(b) > maxManifestBytes {
		return p, errJSONTooLarge
	}
	return p, json.Unmarshal(b, v)
}

type updateManifest struct {
	Schema        string  `json:"schema"`
	UpdateID      string  `json:"update_id"`
	BaseVersionID string  `json:"base_version_id"`
	InnerSteps    int     `json:"inner_steps"`
	Samples       int     `json:"samples"`
	LossFirst     float64 `json:"loss_first"`
	LossLast      float64 `json:"loss_last"`
	LossBefore    float64 `json:"loss_before"`
	LossAfter     float64 `json:"loss_after"`
	DeltaL2       float64 `json:"delta_l2"`
}

// validateLocalTrain verifies a DiLoCo update: it must target the run's
// current global version, have the expected manifest, and pass the Python
// structural/numerical checks (shapes, finiteness, norm bounds).
// Note: this cannot prove the update reduces loss on real data; workers'
// reported losses are recorded as unverified metadata.
func (s *Service) validateLocalTrain(ctx context.Context, in scheduler.ValidationInput) (scheduler.Verdict, error) {
	if in.TrainingRun == nil {
		return scheduler.Verdict{Reason: "job is not attached to a training run"}, nil
	}
	run, err := s.runByID(ctx, *in.TrainingRun)
	if err != nil {
		return scheduler.Verdict{Reason: "training run missing"}, nil
	}
	if in.ModelVersion == nil || run.ModelVersion == nil || *in.ModelVersion != *run.ModelVersion {
		return scheduler.Verdict{Reason: "stale_base_version: the global model advanced before this update arrived"}, nil
	}
	upd, wts := in.Artifacts["update"], in.Artifacts["update_weights"]
	if upd == "" || wts == "" {
		return scheduler.Verdict{Reason: "missing update or update_weights artifact", Retry: true}, nil
	}
	var params struct {
		UpdateID   string `json:"update_id"`
		InnerSteps int    `json:"inner_steps"`
		BatchSize  int    `json:"batch_size"`
	}
	_ = json.Unmarshal(in.Params, &params)
	var man updateManifest
	manPath, err := s.readJSONArtifact(ctx, in.ProjectID, upd, &man)
	if err != nil {
		return scheduler.Verdict{Reason: "unreadable or oversized update manifest: " + err.Error()}, nil
	}
	if man.UpdateID != params.UpdateID || man.Schema != "nivra.update/1" {
		return scheduler.Verdict{Reason: "update manifest does not match the assigned update_id", Retry: true}, nil
	}
	wPath, wInfo, err := s.store.Path(ctx, in.ProjectID, wts)
	if err != nil {
		return scheduler.Verdict{}, err
	}
	base, err := s.version(ctx, *run.ModelVersion)
	if err != nil {
		return scheduler.Verdict{}, err
	}
	// An honest update has the same tensors as the base weights; reject
	// anything far from that size before Python loads it.
	if _, bInfo, err := s.store.Path(ctx, in.ProjectID, base.Weights); err == nil {
		if d := wInfo.Size - bInfo.Size; d > 1<<20 || d < -(1<<20) {
			return scheduler.Verdict{Reason: fmt.Sprintf("update_weights size %d is inconsistent with the base weights (%d)", wInfo.Size, bInfo.Size)}, nil
		}
	}
	if man.BaseVersionID != base.VersionID {
		return scheduler.Verdict{Reason: "stale_base_version"}, nil
	}
	peers := s.peerNorms(ctx, run.ID, run.Round, in.AttemptID)
	baseW, _, err := s.store.Path(ctx, in.ProjectID, base.Weights)
	if err != nil {
		return scheduler.Verdict{}, err
	}
	baseM, _, err := s.store.Path(ctx, in.ProjectID, base.Manifest)
	if err != nil {
		return scheduler.Verdict{}, err
	}
	params2 := map[string]any{"expected_base_version_id": base.VersionID}
	if len(peers) > 0 {
		params2["peer_norms"] = peers
	}
	res, _, cleanup, err := s.runPython(ctx, "ai.validate_update", run.ID, params2, map[string]any{
		"update": wPath, "update_manifest": manPath, "base_manifest": baseM, "base_weights": baseW})
	if err != nil {
		return scheduler.Verdict{}, err // infrastructure problem: retried later
	}
	defer cleanup()
	var sum struct {
		Valid   bool     `json:"valid"`
		Reasons []string `json:"reasons"`
		L2      float64  `json:"l2_norm"`
	}
	if err := json.Unmarshal(res.Summary, &sum); err != nil {
		return scheduler.Verdict{}, fmt.Errorf("validate_update summary: %w", err)
	}
	detail := map[string]any{"l2_norm": sum.L2, "reasons": sum.Reasons, "update_id": man.UpdateID, "base_version_id": man.BaseVersionID,
		"samples": man.Samples, "loss_first": man.LossFirst, "loss_last": man.LossLast, "loss_before_reported": man.LossBefore,
		"loss_after_reported": man.LossAfter, "worker_losses_verified": false}
	if !sum.Valid {
		return scheduler.Verdict{Reason: fmt.Sprintf("update rejected by validation: %v", sum.Reasons), Detail: detail, Retry: true}, nil
	}
	// Credit derives from the server-issued job parameters, never from
	// numbers in the worker-written manifest.
	samples := float64(params.InnerSteps * params.BatchSize)
	return scheduler.Verdict{Accept: true, Units: samples, UnitKind: "training_samples",
		Credit: samples / 100, Detail: detail}, nil
}

// peerNorms returns the L2 norms of other valid updates in the same round.
func (s *Service) peerNorms(ctx context.Context, runID string, round int, exceptAttempt string) []float64 {
	rows, err := s.d.DB.Query(ctx, `SELECT (r.validation_detail->>'l2_norm')::float8 FROM job_results r JOIN jobs j ON j.id=r.job_id
		WHERE j.training_run_id=$1 AND j.idempotency_key LIKE $2 AND r.validation_status='accepted' AND r.attempt_id <> $3
		AND r.validation_detail ? 'l2_norm'`, runID, roundKeyPrefix(runID, round)+"%", exceptAttempt)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var out []float64
	for rows.Next() {
		var v float64
		if rows.Scan(&v) == nil {
			out = append(out, v)
		}
	}
	return out
}

// validateDatasetShard recomputes small shards on the controller and
// compares the content hash, giving fully verified preprocessing credit.
func (s *Service) validateDatasetShard(ctx context.Context, in scheduler.ValidationInput) (scheduler.Verdict, error) {
	var p struct {
		Seed   int `json:"seed"`
		Start  int `json:"start"`
		Count  int `json:"count"`
		Frames int `json:"frames"`
		Size   int `json:"size"`
	}
	if err := json.Unmarshal(in.Params, &p); err != nil {
		return scheduler.Verdict{Reason: "bad params"}, nil
	}
	var sum struct {
		Count         int    `json:"count"`
		ContentSHA256 string `json:"content_sha256"`
		Fingerprint   string `json:"fingerprint"`
	}
	if err := json.Unmarshal(in.Result, &sum); err != nil || sum.ContentSHA256 == "" {
		return scheduler.Verdict{Reason: "result lacks content_sha256", Retry: true}, nil
	}
	if in.Artifacts["dataset_shard"] == "" {
		return scheduler.Verdict{Reason: "missing dataset_shard artifact", Retry: true}, nil
	}
	if p.Count > 256 {
		return scheduler.Verdict{Accept: true, Units: float64(sum.Count), UnitKind: "clips_unverified", Credit: 0,
			Detail: map[string]any{"verified": false, "note": "shards over 256 clips are not recomputed by the controller"}}, nil
	}
	res, _, cleanup, err := s.runPython(ctx, "ai.dataset_shard", "", json.RawMessage(in.Params), nil)
	if err != nil {
		return scheduler.Verdict{}, err
	}
	defer cleanup()
	var mine struct {
		ContentSHA256 string `json:"content_sha256"`
	}
	if err := json.Unmarshal(res.Summary, &mine); err != nil {
		return scheduler.Verdict{}, err
	}
	if mine.ContentSHA256 != sum.ContentSHA256 {
		return scheduler.Verdict{Reason: "content hash differs from controller recomputation", Retry: true}, nil
	}
	// The uploaded shard file itself must equal the controller's recomputation.
	ref, ok := artifactByKind(res, "dataset_shard")
	if !ok || ref.SHA256 == "" {
		return scheduler.Verdict{}, errors.New("controller recomputation produced no dataset_shard artifact hash")
	}
	upInfo, err := s.store.Get(ctx, in.ProjectID, in.Artifacts["dataset_shard"])
	if err != nil {
		return scheduler.Verdict{Reason: "dataset_shard artifact missing", Retry: true}, nil
	}
	if upInfo.SHA256 != ref.SHA256 {
		return scheduler.Verdict{Reason: "uploaded shard file differs from controller recomputation", Retry: true}, nil
	}
	return scheduler.Verdict{Accept: true, Units: float64(p.Count), UnitKind: "clips", Credit: float64(p.Count) / 100,
		Detail: map[string]any{"verified": true, "content_sha256": mine.ContentSHA256}}, nil
}

// validateEvaluate accepts structurally valid evaluation reports. Their
// numbers are not independently recomputed, so they earn no credit.
func (s *Service) validateEvaluate(ctx context.Context, in scheduler.ValidationInput) (scheduler.Verdict, error) {
	if in.Artifacts["evaluation_report"] == "" {
		return scheduler.Verdict{Reason: "missing evaluation_report artifact", Retry: true}, nil
	}
	var rep map[string]any
	if _, err := s.readJSONArtifact(ctx, in.ProjectID, in.Artifacts["evaluation_report"], &rep); err != nil {
		return scheduler.Verdict{Reason: "unreadable or oversized evaluation report"}, nil
	}
	return scheduler.Verdict{Accept: true, Units: 1, UnitKind: "evaluations", Credit: 0,
		Detail: map[string]any{"verified": false}}, nil
}

// validateGenerate accepts a generation if an MP4 artifact was produced.
func (s *Service) validateGenerate(ctx context.Context, in scheduler.ValidationInput) (scheduler.Verdict, error) {
	return s.validateMP4(ctx, in, "video")
}

// mp4Validator accepts a result whose artifact of the given kind is an MP4.
func mp4Validator(kind string) func(*Service, context.Context, scheduler.ValidationInput) (scheduler.Verdict, error) {
	return func(s *Service, ctx context.Context, in scheduler.ValidationInput) (scheduler.Verdict, error) {
		return s.validateMP4(ctx, in, kind)
	}
}

func (s *Service) validateMP4(ctx context.Context, in scheduler.ValidationInput, kind string) (scheduler.Verdict, error) {
	id := in.Artifacts[kind]
	if id == "" {
		return scheduler.Verdict{Reason: "no " + kind + " artifact produced", Retry: true}, nil
	}
	info, err := s.store.Get(ctx, in.ProjectID, id)
	if err != nil || info.Size < 100 {
		return scheduler.Verdict{Reason: "video artifact missing or empty", Retry: true}, nil
	}
	f, err := os.Open(mustPath(s, ctx, in.ProjectID, id))
	if err != nil {
		return scheduler.Verdict{}, err
	}
	defer f.Close()
	head := make([]byte, 12)
	if _, err := io.ReadFull(f, head); err != nil || string(head[4:8]) != "ftyp" {
		return scheduler.Verdict{Reason: "artifact is not an MP4 file"}, nil
	}
	return scheduler.Verdict{Accept: true, Units: 1, UnitKind: "videos", Credit: 0}, nil
}

func mustPath(s *Service, ctx context.Context, projectID, id string) string {
	p, _, err := s.store.Path(ctx, projectID, id)
	if err != nil {
		return ""
	}
	return p
}

var errNoVersion = errors.New("model version not found")
