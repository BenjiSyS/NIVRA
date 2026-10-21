package training

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"sync"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"nivra/internal/controller/core"
	"nivra/internal/controller/scheduler"
)

// Validators for the HD job types. Everything here runs on the controller
// without loading a model: safetensors headers are parsed directly and the fp16
// delta is streamed once, tensor by tensor.

func registerHDJobTypes(reg *scheduler.Registry, svc *Service) {
	reg.Register(&scheduler.JobType{Name: "ai.hd_encode", Category: "preprocessing", Async: true,
		ValidateParams: objectParams, Validate: svcOr(svc, (*Service).validateHDEncode), OnFinalized: svcKick(svc)})
	reg.Register(&scheduler.JobType{Name: "ai.hd_train", Category: "training", Async: true,
		ValidateParams: objectParams, Validate: svcOr(svc, (*Service).validateHDTrain), OnFinalized: svcKick(svc)})
	reg.Register(&scheduler.JobType{Name: "ai.hd_eval", Category: "evaluation", Async: true,
		ValidateParams: objectParams, Validate: svcOr(svc, (*Service).validateHDEval), OnFinalized: svcKick(svc)})
}

// ---- safetensors ----

type stTensor struct {
	DType string
	Shape []int64
	Begin int64
	End   int64
}

func (t stTensor) numel() int64 {
	n := int64(1)
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// stHeader is a parsed, layout-checked safetensors header.
type stHeader struct {
	Tensors   map[string]stTensor
	Order     []string // by data offset
	DataStart int64
	DataLen   int64
	Size      int64
}

const maxSafetensorsHeader = 64 << 20

// readSafetensorsHeader parses the header of a safetensors file and checks that
// the tensors tile the data area exactly (no gaps, overlaps or trailing bytes).
func readSafetensorsHeader(path string) (*stHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, errors.New("not a safetensors file (too short)")
	}
	hl := binary.LittleEndian.Uint64(lenBuf[:])
	if hl == 0 || hl > maxSafetensorsHeader || int64(hl) > fi.Size()-8 {
		return nil, errors.New("not a safetensors file (bad header length)")
	}
	raw := make([]byte, hl)
	if _, err := io.ReadFull(f, raw); err != nil {
		return nil, errors.New("truncated safetensors header")
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("safetensors header is not JSON: %v", err)
	}
	h := &stHeader{Tensors: map[string]stTensor{}, DataStart: 8 + int64(hl), Size: fi.Size()}
	h.DataLen = h.Size - h.DataStart
	for name, v := range top {
		if name == "__metadata__" {
			continue
		}
		var e struct {
			DType   string  `json:"dtype"`
			Shape   []int64 `json:"shape"`
			Offsets []int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(v, &e); err != nil || len(e.Offsets) != 2 {
			return nil, fmt.Errorf("tensor %q has a malformed header entry", name)
		}
		t := stTensor{DType: e.DType, Shape: e.Shape, Begin: e.Offsets[0], End: e.Offsets[1]}
		n := int64(1)
		for _, d := range t.Shape {
			if d < 0 || d > 1<<40 || (d != 0 && n > (1<<50)/d) {
				return nil, fmt.Errorf("tensor %q has an invalid shape", name)
			}
			n *= d
		}
		if t.Begin < 0 || t.End < t.Begin {
			return nil, fmt.Errorf("tensor %q has invalid data offsets", name)
		}
		h.Tensors[name] = t
		h.Order = append(h.Order, name)
	}
	sort.Slice(h.Order, func(a, b int) bool {
		ta, tb := h.Tensors[h.Order[a]], h.Tensors[h.Order[b]]
		if ta.Begin != tb.Begin {
			return ta.Begin < tb.Begin
		}
		return h.Order[a] < h.Order[b]
	})
	var pos int64
	for _, name := range h.Order {
		t := h.Tensors[name]
		if t.Begin != pos {
			return nil, fmt.Errorf("tensor %q does not start where the previous one ends", name)
		}
		pos = t.End
	}
	if pos != h.DataLen {
		return nil, fmt.Errorf("data area is %d bytes but the tensors cover %d", h.DataLen, pos)
	}
	return h, nil
}

// checkDeltaStructure verifies the update is fp16, the same tensors and shapes
// as the base, and exactly sum(numel)*2 + header bytes long.
func checkDeltaStructure(upd, base *stHeader) string {
	if len(upd.Tensors) != len(base.Tensors) {
		return fmt.Sprintf("update has %d tensors, the base has %d", len(upd.Tensors), len(base.Tensors))
	}
	var want int64
	for name, bt := range base.Tensors {
		ut, ok := upd.Tensors[name]
		if !ok {
			return fmt.Sprintf("update lacks tensor %q", name)
		}
		if ut.DType != "F16" {
			return fmt.Sprintf("tensor %q is %s, deltas must be F16", name, ut.DType)
		}
		if len(ut.Shape) != len(bt.Shape) {
			return fmt.Sprintf("tensor %q has the wrong rank", name)
		}
		for i := range ut.Shape {
			if ut.Shape[i] != bt.Shape[i] {
				return fmt.Sprintf("tensor %q has shape %v, the base has %v", name, ut.Shape, bt.Shape)
			}
		}
		if ut.End-ut.Begin != ut.numel()*2 {
			return fmt.Sprintf("tensor %q holds %d bytes for %d elements", name, ut.End-ut.Begin, ut.numel())
		}
		want += ut.numel() * 2
	}
	if upd.Size != upd.DataStart+want {
		return fmt.Sprintf("update file is %d bytes, expected exactly %d", upd.Size, upd.DataStart+want)
	}
	return ""
}

var (
	f16Once sync.Once
	f16Tab  [65536]float32
)

func f16Table() *[65536]float32 {
	f16Once.Do(func() {
		for i := 0; i < 65536; i++ {
			sign := float64(1)
			if i&0x8000 != 0 {
				sign = -1
			}
			e, m := (i>>10)&0x1f, i&0x3ff
			switch {
			case e == 0:
				f16Tab[i] = float32(sign * math.Ldexp(float64(m), -24))
			case e == 31:
				f16Tab[i] = float32(math.NaN())
			default:
				f16Tab[i] = float32(sign * math.Ldexp(1+float64(m)/1024, e-15))
			}
		}
	})
	return &f16Tab
}

// deltaStats is what the streaming scan found.
type deltaStats struct {
	L2      float64
	Linf    float64
	Nonzero bool
}

// scanDelta streams the fp16 data once. It rejects NaN/Inf, a tensor whose
// ||d||_inf exceeds linfMax or ||d||_2 exceeds linfMax*sqrt(numel), and an
// all-zero update (docs/architecture/HD_TRAINING.md, safety constraints).
func scanDelta(path string, h *stHeader, linfMax float64) (deltaStats, string, error) {
	var st deltaStats
	f, err := os.Open(path)
	if err != nil {
		return st, "", err
	}
	defer f.Close()
	if _, err := f.Seek(h.DataStart, io.SeekStart); err != nil {
		return st, "", err
	}
	br := bufio.NewReaderSize(f, 1<<20)
	tab := f16Table()
	buf := make([]byte, 1<<20)
	var total float64
	for _, name := range h.Order {
		t := h.Tensors[name]
		left := t.End - t.Begin
		var sumsq, linf float64
		for left > 0 {
			n := int64(len(buf))
			if left < n {
				n = left
			}
			if _, err := io.ReadFull(br, buf[:n]); err != nil {
				return st, "", err
			}
			for i := int64(0); i+1 < n; i += 2 {
				u := uint16(buf[i]) | uint16(buf[i+1])<<8
				if u&0x7c00 == 0x7c00 {
					return st, fmt.Sprintf("tensor %q contains NaN or Inf", name), nil
				}
				v := float64(tab[u])
				if v != 0 {
					st.Nonzero = true
					sumsq += v * v
					if a := math.Abs(v); a > linf {
						linf = a
					}
				}
			}
			left -= n
		}
		l2 := math.Sqrt(sumsq)
		if linf > linfMax {
			return st, fmt.Sprintf("tensor %q: max |delta| %.4g exceeds the bound %.4g", name, linf, linfMax), nil
		}
		if n := t.numel(); n > 0 && l2 > linfMax*math.Sqrt(float64(n)) {
			return st, fmt.Sprintf("tensor %q: delta L2 %.4g exceeds the bound %.4g", name, l2, linfMax*math.Sqrt(float64(n))), nil
		}
		total += sumsq
		if linf > st.Linf {
			st.Linf = linf
		}
	}
	st.L2 = math.Sqrt(total)
	if !st.Nonzero {
		return st, "update is all zeros", nil
	}
	return st, "", nil
}

// ---- ai.hd_train ----

type hdUpdateManifest struct {
	Schema        string     `json:"schema"`
	RunID         string     `json:"run_id"`
	Round         int        `json:"round"`
	RoundBatch    int        `json:"round_batch"`
	BaseVersionID string     `json:"base_version_id"`
	UpdateID      string     `json:"update_id"`
	AttemptID     string     `json:"attempt_id"`
	Target        string     `json:"target"`
	HDConfigSHA   string     `json:"hd_config_sha"`
	Stage         Stage      `json:"stage"`
	InnerSteps    int        `json:"inner_steps"`
	LR            float64    `json:"lr"`
	Shards        []shardRef `json:"shards"`
	SamplesSeen   int64      `json:"samples_seen"`
	DeltaDtype    string     `json:"delta_dtype"`
}

// hdEchoMismatch compares the manifest the worker wrote with the params the
// controller issued and the attempt that delivered it. "" = consistent.
func hdEchoMismatch(p hdTrainParams, m hdUpdateManifest, attemptID string) string {
	diff := func(field string) string { return "update manifest " + field + " does not match the job" }
	switch {
	case len(m.Schema) < 14 || m.Schema[:13] != "nivra.update/":
		return "update manifest has no nivra.update/N schema"
	case m.RunID != p.RunID:
		return diff("run_id")
	case m.Round != p.Round:
		return diff("round")
	case m.RoundBatch != p.RoundBatch:
		return diff("round_batch")
	case m.BaseVersionID != p.BaseVersionID:
		return diff("base_version_id")
	case m.UpdateID != p.UpdateID:
		return diff("update_id")
	case m.AttemptID != attemptID:
		return diff("attempt_id")
	case m.Target != p.Target:
		return diff("target")
	case m.HDConfigSHA != p.HDConfigSHA:
		return diff("hd_config_sha")
	case m.Stage != p.Stage:
		return diff("stage")
	case m.InnerSteps != p.InnerSteps:
		return diff("inner_steps")
	case math.Abs(m.LR-p.LR) > 1e-6*math.Max(math.Abs(p.LR), 1e-12):
		return diff("lr")
	case m.DeltaDtype != p.DeltaDtype:
		return diff("delta_dtype")
	case len(m.Shards) != len(p.Shards):
		return diff("shards")
	}
	for i := range p.Shards {
		if m.Shards[i] != p.Shards[i] {
			return diff("shards")
		}
	}
	if max := int64(p.InnerSteps) * int64(p.BatchSize) * int64(p.GradAccum); m.SamplesSeen < 1 || m.SamplesSeen > max {
		return fmt.Sprintf("update manifest samples_seen %d is outside 1..%d", m.SamplesSeen, max)
	}
	return ""
}

// claimHDUpdate records the update's content hash. The unique index on
// (run_id, round, content_sha256) makes a replayed delta lose the race.
func (s *Service) claimHDUpdate(ctx context.Context, in scheduler.ValidationInput, p hdTrainParams, baseID, updateArtifact, contentSHA string, detail map[string]any) (dup bool, err error) {
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		// A stale claim of an earlier attempt of the same job must not block this one.
		if _, err := tx.Exec(ctx, `DELETE FROM training_updates WHERE run_id=$1 AND update_id=$2 AND attempt_id<>$3 AND state='pending'`,
			p.RunID, p.UpdateID, in.AttemptID); err != nil {
			return err
		}
		_, err := tx.Exec(ctx, `INSERT INTO training_updates (run_id, update_id, job_id, attempt_id, device_id, base_version, round, update_artifact,
				metadata, state, content_sha256) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'pending',$10)
			ON CONFLICT (attempt_id) DO UPDATE SET content_sha256=EXCLUDED.content_sha256`,
			p.RunID, p.UpdateID, in.JobID, in.AttemptID, in.DeviceID, baseID, p.Round, nilIfEmpty(updateArtifact), toJSON(detail), contentSHA)
		return err
	})
	var pe *pgconn.PgError
	if errors.As(err, &pe) && pe.Code == "23505" && pe.ConstraintName == "training_updates_content_uq" {
		return true, nil
	}
	return false, err
}

// validateHDTrain accepts an ai.hd_train result only if it is bound to this
// exact job (echo), has exactly the size and structure of an fp16 delta of the
// base, passes the numerical bounds and is not a replay of another update.
func (s *Service) validateHDTrain(ctx context.Context, in scheduler.ValidationInput) (scheduler.Verdict, error) {
	if in.TrainingRun == nil {
		return scheduler.Verdict{Reason: "job is not attached to a training run"}, nil
	}
	run, err := s.runByID(ctx, *in.TrainingRun)
	if err != nil {
		return scheduler.Verdict{Reason: "training run missing"}, nil
	}
	var p hdTrainParams
	if err := json.Unmarshal(in.Params, &p); err != nil {
		return scheduler.Verdict{Reason: "bad job params"}, nil
	}
	if in.ModelVersion == nil || run.ModelVersion == nil || *in.ModelVersion != *run.ModelVersion || p.Round != run.Round {
		return scheduler.Verdict{Reason: "stale_base_version: the global model advanced before this update arrived"}, nil
	}
	upd, wts := in.Artifacts["update"], in.Artifacts["update_weights"]
	if upd == "" || wts == "" {
		return scheduler.Verdict{Reason: "missing update or update_weights artifact", Retry: true}, nil
	}
	var man hdUpdateManifest
	if _, err := s.readJSONArtifact(ctx, in.ProjectID, upd, &man); err != nil {
		return scheduler.Verdict{Reason: "unreadable or oversized update manifest: " + err.Error(), Retry: true}, nil
	}
	if why := hdEchoMismatch(p, man, in.AttemptID); why != "" {
		return scheduler.Verdict{Reason: why, Retry: true}, nil
	}
	base, err := s.hdVersionRow(ctx, run.ProjectID, *run.ModelVersion)
	if err != nil || base.trainOrMaster() == nil {
		return scheduler.Verdict{}, fmt.Errorf("base version unavailable: %v", err)
	}
	wPath, wInfo, err := s.store.Path(ctx, in.ProjectID, wts)
	if err != nil {
		return scheduler.Verdict{}, err
	}
	bPath, _, err := s.store.Path(ctx, in.ProjectID, *base.trainOrMaster())
	if err != nil {
		return scheduler.Verdict{}, err
	}
	uh, err := readSafetensorsHeader(wPath)
	if err != nil {
		return scheduler.Verdict{Reason: "update_weights: " + err.Error(), Retry: true}, nil
	}
	bh, err := readSafetensorsHeader(bPath)
	if err != nil {
		return scheduler.Verdict{}, fmt.Errorf("base weights: %w", err)
	}
	if why := checkDeltaStructure(uh, bh); why != "" {
		return scheduler.Verdict{Reason: "update_weights rejected: " + why, Retry: true}, nil
	}
	c := run.Config
	stats, why, err := scanDelta(wPath, uh, c.BoundC*float64(p.InnerSteps)*p.LR)
	if err != nil {
		return scheduler.Verdict{}, err
	}
	if why != "" {
		return scheduler.Verdict{Reason: "update_weights rejected: " + why, Retry: true}, nil
	}
	var res struct {
		LossFirst  float64 `json:"loss_first"`
		LossLast   float64 `json:"loss_last"`
		DeviceCls  string  `json:"device_class"`
		SamplesSee float64 `json:"samples_seen"`
	}
	_ = json.Unmarshal(in.Result, &res)
	detail := map[string]any{"update_id": p.UpdateID, "base_version_id": p.BaseVersionID, "l2_norm": stats.L2, "linf": stats.Linf,
		"samples_seen": float64(man.SamplesSeen), "loss_first": res.LossFirst, "loss_last": res.LossLast, "device_class": res.DeviceCls,
		"content_sha256": wInfo.SHA256, "worker_losses_verified": false}
	dup, err := s.claimHDUpdate(ctx, in, p, base.ID, upd, wInfo.SHA256, detail)
	if err != nil {
		return scheduler.Verdict{}, err
	}
	if dup {
		return scheduler.Verdict{Reason: "duplicate update_weights content in this round", Detail: detail}, nil
	}
	samples := float64(p.InnerSteps * p.BatchSize * p.GradAccum)
	return scheduler.Verdict{Accept: true, Units: samples, UnitKind: "training_samples", Credit: samples / 100, Detail: detail}, nil
}

// ---- ai.hd_encode ----

type hdEncodeParams struct {
	DatasetID string `json:"dataset_id"`
	Target    string `json:"target_dataset_id"`
	ShardIdx  int    `json:"shard_idx"`
	ShardSHA  string `json:"shard_sha256"`
	Audit     bool   `json:"audit"`
}

func finite(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) }

func (s *Service) validateHDEncode(ctx context.Context, in scheduler.ValidationInput) (scheduler.Verdict, error) {
	var p hdEncodeParams
	if err := json.Unmarshal(in.Params, &p); err != nil || !uuidRe.MatchString(p.DatasetID) || !uuidRe.MatchString(p.Target) {
		return scheduler.Verdict{Reason: "bad job params"}, nil
	}
	art := in.Artifacts["latent_shard"]
	if art == "" {
		return scheduler.Verdict{Reason: "missing latent_shard artifact", Retry: true}, nil
	}
	var sum struct {
		Clips      int     `json:"clips"`
		SHA256     string  `json:"sha256"`
		LatentMean float64 `json:"latent_mean"`
		LatentStd  float64 `json:"latent_std"`
	}
	if err := json.Unmarshal(in.Result, &sum); err != nil {
		return scheduler.Verdict{Reason: "result summary is not valid", Retry: true}, nil
	}
	info, err := s.store.Get(ctx, in.ProjectID, art)
	if err != nil || info.State != "complete" || info.Size == 0 {
		return scheduler.Verdict{Reason: "latent_shard artifact missing or empty", Retry: true}, nil
	}
	if sum.SHA256 != info.SHA256 {
		return scheduler.Verdict{Reason: "summary sha256 does not match the uploaded latent_shard", Retry: true}, nil
	}
	if !finite(sum.LatentMean) || !finite(sum.LatentStd) || sum.LatentStd <= 0 {
		return scheduler.Verdict{Reason: "latent statistics are not finite and positive (degenerate encode)", Retry: true}, nil
	}
	var rawClips int
	var rawSHA string
	if err := s.d.DB.QueryRow(ctx, `SELECT clips, sha256 FROM dataset_shards WHERE dataset_id=$1 AND idx=$2`, p.DatasetID, p.ShardIdx).Scan(&rawClips, &rawSHA); err != nil {
		return scheduler.Verdict{Reason: "the raw shard is not part of the dataset"}, nil
	}
	if rawSHA != p.ShardSHA {
		return scheduler.Verdict{Reason: "job refers to a different raw shard than the dataset registry"}, nil
	}
	if sum.Clips != rawClips {
		return scheduler.Verdict{Reason: fmt.Sprintf("encoded %d clips, the raw shard has %d", sum.Clips, rawClips), Retry: true}, nil
	}
	if p.Audit {
		var dev string
		err := s.d.DB.QueryRow(ctx, `SELECT a.device_id::text FROM dataset_jobs d JOIN job_results r ON r.job_id=d.job_id AND r.validation_status='accepted'
			JOIN job_attempts a ON a.id=r.attempt_id WHERE d.dataset_id=$1 AND d.kind='encode' AND d.idx=$2`, p.Target, p.ShardIdx).Scan(&dev)
		if err == nil && dev == in.DeviceID {
			return scheduler.Verdict{Reason: "an audit re-encode must run on a different device than the first encode", Retry: true}, nil
		}
	}
	return scheduler.Verdict{Accept: true, Units: float64(sum.Clips), UnitKind: "clips_encoded", Credit: 0,
		Detail: map[string]any{"clips": float64(sum.Clips), "latent_mean": sum.LatentMean, "latent_std": sum.LatentStd,
			"sha256": info.SHA256, "device_id": in.DeviceID, "verified": false}}, nil
}

// ---- ai.hd_eval ----

func (s *Service) validateHDEval(ctx context.Context, in scheduler.ValidationInput) (scheduler.Verdict, error) {
	var p struct {
		VersionID string `json:"version_id"`
	}
	_ = json.Unmarshal(in.Params, &p)
	art := in.Artifacts["eval_report"]
	if art == "" {
		return scheduler.Verdict{Reason: "missing eval_report artifact", Retry: true}, nil
	}
	var rep struct {
		VersionID string  `json:"version_id"`
		Loss      float64 `json:"loss"`
		N         int     `json:"n"`
	}
	if _, err := s.readJSONArtifact(ctx, in.ProjectID, art, &rep); err != nil {
		return scheduler.Verdict{Reason: "unreadable or oversized eval report", Retry: true}, nil
	}
	if p.VersionID == "" || rep.VersionID != p.VersionID {
		return scheduler.Verdict{Reason: "eval report is for a different version", Retry: true}, nil
	}
	if !finite(rep.Loss) || rep.Loss < 0 || rep.N < 1 {
		return scheduler.Verdict{Reason: "eval report has an invalid loss or sample count", Retry: true}, nil
	}
	return scheduler.Verdict{Accept: true, Units: float64(rep.N), UnitKind: "eval_triples", Credit: 0,
		Detail: map[string]any{"loss": rep.Loss, "n": float64(rep.N), "version_id": rep.VersionID, "verified": false}}, nil
}
