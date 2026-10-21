package training

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"

	"nivra/internal/controller/api"
	"nivra/internal/controller/artifacts"
	"nivra/internal/controller/core"
	"nivra/internal/controller/scheduler"
	"nivra/internal/shared/protocol"
)

// Dataset registry (docs/architecture/HD_TRAINING.md): shards are uploaded as
// content-addressed artifacts, then registered; an encode pass turns a raw
// dataset into a latents dataset bound to a frozen VAE and text encoder.

const (
	maxDatasetShards = 20000
	defaultAuditFrac = 0.05
)

var fingerprintRe = regexp.MustCompile(`^[0-9A-Za-z:_.-]{8,128}$`)

// datasetEncoder is what a latents dataset is bound to.
type datasetEncoder struct {
	VAEVersionID        string `json:"vae_version_id,omitempty"`
	TextEncoderID       string `json:"text_encoder_id,omitempty"`
	TextEncoderArtifact string `json:"text_encoder_artifact,omitempty"`
	TokenizerArtifact   string `json:"tokenizer_artifact,omitempty"`
	Stage               *Stage `json:"stage,omitempty"`
}

type datasetRow struct {
	ID             string         `json:"id"`
	ProjectID      string         `json:"project_id"`
	Name           string         `json:"name"`
	Kind           string         `json:"kind"`
	ParentID       *string        `json:"parent_id"`
	Fingerprint    string         `json:"fingerprint"`
	Encoder        datasetEncoder `json:"encoder"`
	Redistribution string         `json:"redistribution"`
	Status         string         `json:"status"`
	Detail         string         `json:"detail"`
	ClipCount      int64          `json:"clip_count"`
	CreatedAt      string         `json:"created_at"`
}

type datasetShard struct {
	Idx        int    `json:"idx"`
	Split      string `json:"split"`
	ArtifactID string `json:"artifact_id"`
	SHA256     string `json:"sha256"`
	Clips      int    `json:"clips"`
	Bytes      int64  `json:"bytes"`
}

const datasetCols = `id::text, project_id::text, name, kind, parent_id::text, fingerprint, encoder, redistribution, status, detail, clip_count, created_at::text`

func scanDataset(row pgx.Row) (*datasetRow, error) {
	d := &datasetRow{}
	var enc []byte
	err := row.Scan(&d.ID, &d.ProjectID, &d.Name, &d.Kind, &d.ParentID, &d.Fingerprint, &enc, &d.Redistribution, &d.Status, &d.Detail, &d.ClipCount, &d.CreatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	_ = json.Unmarshal(enc, &d.Encoder)
	return d, nil
}

func (s *Service) getDataset(ctx context.Context, projectID, id string) (*datasetRow, error) {
	if !uuidRe.MatchString(id) {
		return nil, ErrNotFound
	}
	return scanDataset(s.d.DB.QueryRow(ctx, `SELECT `+datasetCols+` FROM datasets WHERE id=$1 AND project_id=$2`, id, projectID))
}

func (s *Service) datasetShards(ctx context.Context, q core.Querier, id string) ([]datasetShard, error) {
	rows, err := q.Query(ctx, `SELECT idx, split, artifact_id::text, sha256, clips, bytes FROM dataset_shards WHERE dataset_id=$1 ORDER BY idx`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []datasetShard
	for rows.Next() {
		var x datasetShard
		if err := rows.Scan(&x.Idx, &x.Split, &x.ArtifactID, &x.SHA256, &x.Clips, &x.Bytes); err != nil {
			return nil, err
		}
		out = append(out, x)
	}
	return out, rows.Err()
}

func splitShards(all []datasetShard) (train, val []datasetShard) {
	for _, s := range all {
		if s.Split == "val" {
			val = append(val, s)
		} else {
			train = append(train, s)
		}
	}
	return
}

// datasetSplits loads a dataset's train and val shards (ordered by idx).
func (s *Service) datasetSplits(ctx context.Context, projectID, id string) (train, val []datasetShard, err error) {
	ds, err := s.getDataset(ctx, projectID, id)
	if err != nil {
		return nil, nil, err
	}
	all, err := s.datasetShards(ctx, s.d.DB, ds.ID)
	if err != nil {
		return nil, nil, err
	}
	train, val = splitShards(all)
	return
}

// evalShards picks the val shards used for held-out evaluation: in idx order
// until they hold at least n clips, at most 4.
func evalShards(val []datasetShard, n int) []datasetShard {
	var out []datasetShard
	clips := 0
	for _, v := range val {
		if len(out) >= 4 || clips >= n {
			break
		}
		out = append(out, v)
		clips += v.Clips
	}
	return out
}

// shardsFingerprint is the content identity of a shard list.
func shardsFingerprint(kind string, sh []datasetShard) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\n", kind)
	for _, x := range sh {
		fmt.Fprintf(h, "%d:%s:%s\n", x.Idx, x.Split, x.SHA256)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ---- registration ----

type registerRequest struct {
	Name           string         `json:"name"`
	Kind           string         `json:"kind"`
	Fingerprint    string         `json:"fingerprint"`
	ParentID       string         `json:"parent_dataset_id"`
	Encoder        datasetEncoder `json:"encoder"`
	Redistribution string         `json:"redistribution"`
	Shards         []datasetShard `json:"shards"`
}

func badReq(f string, a ...any) error {
	return fmt.Errorf("%w: %s", ErrBadConfig, fmt.Sprintf(f, a...))
}

// RegisterDataset verifies every referenced artifact (exists in this project,
// complete, right kind, sha256 as declared) and stores the dataset. Registering
// the same content twice is a conflict.
func (s *Service) RegisterDataset(ctx context.Context, actor core.Actor, projectID string, req registerRequest) (*datasetRow, error) {
	req.Name = strings.TrimSpace(req.Name)
	if req.Name == "" || len(req.Name) > 80 {
		return nil, badReq("name must be 1-80 characters")
	}
	if req.Kind != "raw" && req.Kind != "latents" {
		return nil, badReq("kind must be raw or latents")
	}
	if req.Redistribution == "" {
		req.Redistribution = "owner_devices"
	}
	if req.Redistribution != "owner_devices" && req.Redistribution != "any" {
		return nil, badReq("redistribution must be owner_devices or any")
	}
	if n := len(req.Shards); n < 1 || n > maxDatasetShards {
		return nil, badReq("1..%d shards required", maxDatasetShards)
	}
	wantKind := map[string]string{"raw": "dataset_shard", "latents": "latent_shard"}[req.Kind]
	seenIdx, seenSHA, seenArt := map[int]bool{}, map[string]bool{}, map[string]bool{}
	var clips int64
	train := 0
	for i := range req.Shards {
		sh := &req.Shards[i]
		sh.SHA256 = strings.ToLower(sh.SHA256)
		switch {
		case sh.Idx < 0:
			return nil, badReq("shard idx must be >= 0")
		case sh.Split != "train" && sh.Split != "val":
			return nil, badReq("shard %d: split must be train or val", sh.Idx)
		case sh.Clips < 1:
			return nil, badReq("shard %d: clips must be >= 1", sh.Idx)
		case seenIdx[sh.Idx]:
			return nil, fmt.Errorf("%w: duplicate shard idx %d", ErrBadState, sh.Idx)
		case seenSHA[sh.SHA256]:
			return nil, fmt.Errorf("%w: shard %d repeats the content of another shard", ErrBadState, sh.Idx)
		case seenArt[sh.ArtifactID]:
			return nil, fmt.Errorf("%w: shard %d reuses an artifact", ErrBadState, sh.Idx)
		}
		info, err := s.store.Get(ctx, projectID, sh.ArtifactID)
		if err != nil {
			return nil, badReq("shard %d: artifact not found in this project", sh.Idx)
		}
		switch {
		case info.State != "complete":
			return nil, badReq("shard %d: artifact upload is not complete", sh.Idx)
		case info.Kind != wantKind:
			return nil, badReq("shard %d: artifact kind is %q, a %s dataset needs %q", sh.Idx, info.Kind, req.Kind, wantKind)
		case info.SHA256 != sh.SHA256:
			return nil, badReq("shard %d: sha256 does not match the stored artifact", sh.Idx)
		case sh.Bytes != 0 && sh.Bytes != info.Size:
			return nil, badReq("shard %d: bytes does not match the stored artifact (%d)", sh.Idx, info.Size)
		}
		sh.ArtifactID, sh.Bytes = info.ID, info.Size
		seenIdx[sh.Idx], seenSHA[sh.SHA256], seenArt[info.ID] = true, true, true
		clips += int64(sh.Clips)
		if sh.Split == "train" {
			train++
		}
	}
	if train == 0 {
		return nil, badReq("at least one train shard is required")
	}
	sort.Slice(req.Shards, func(a, b int) bool { return req.Shards[a].Idx < req.Shards[b].Idx })
	if req.Fingerprint == "" {
		req.Fingerprint = shardsFingerprint(req.Kind, req.Shards)
	}
	if !fingerprintRe.MatchString(req.Fingerprint) {
		return nil, badReq("fingerprint must be 8-128 characters of [0-9A-Za-z:_.-]")
	}
	if req.Kind == "latents" {
		e := req.Encoder
		if e.VAEVersionID == "" || e.TextEncoderID == "" || len(e.TextEncoderID) > 128 || e.Stage == nil {
			return nil, badReq("a latents dataset needs encoder {vae_version_id, text_encoder_id, stage}")
		}
		if _, err := s.hdVersionRow(ctx, projectID, e.VAEVersionID); err != nil {
			return nil, badReq("encoder.vae_version_id is not a model version of this project")
		}
	} else if req.Encoder.VAEVersionID != "" || req.Encoder.TextEncoderID != "" {
		return nil, badReq("a raw dataset has no vae/text encoder binding")
	}
	if st := req.Encoder.Stage; st != nil && (st.H < 8 || st.W < 8 || st.Frames < 1 || st.FPS < 1 || st.H > 4320 || st.W > 7680 || st.Frames > 2000 || st.FPS > 120) {
		return nil, badReq("encoder.stage out of range")
	}
	var parent any
	if req.ParentID != "" {
		if _, err := s.getDataset(ctx, projectID, req.ParentID); err != nil {
			return nil, badReq("parent_dataset_id not found in this project")
		}
		parent = req.ParentID
	}
	var by any
	if actor.Type == "user" {
		by = actor.ID
	}
	enc, _ := json.Marshal(req.Encoder)
	var id string
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		if err := tx.QueryRow(ctx, `INSERT INTO datasets (project_id, name, kind, parent_id, fingerprint, encoder, redistribution, status, clip_count, created_by)
			VALUES ($1,$2,$3,$4,$5,$6,$7,'ready',$8,$9) RETURNING id::text`,
			projectID, req.Name, req.Kind, parent, req.Fingerprint, enc, req.Redistribution, clips, by).Scan(&id); err != nil {
			return err
		}
		for _, sh := range req.Shards {
			if _, err := tx.Exec(ctx, `INSERT INTO dataset_shards (dataset_id, idx, split, artifact_id, sha256, clips, bytes) VALUES ($1,$2,$3,$4,$5,$6,$7)`,
				id, sh.Idx, sh.Split, sh.ArtifactID, sh.SHA256, sh.Clips, sh.Bytes); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `UPDATE artifacts SET protected=true WHERE id=$1`, sh.ArtifactID); err != nil {
				return err
			}
		}
		return nil
	})
	var pe *pgconn.PgError
	if errors.As(err, &pe) && pe.Code == "23505" {
		return nil, fmt.Errorf("%w: a %s dataset with this fingerprint is already registered", ErrBadState, req.Kind)
	}
	if err != nil {
		return nil, err
	}
	s.d.Audit(ctx, nil, actor, "dataset.register", "dataset:"+id, "success", map[string]any{"name": req.Name, "kind": req.Kind, "shards": len(req.Shards)})
	return s.getDataset(ctx, projectID, id)
}

// ---- encode pass ----

type encodeRequest struct {
	VAEVersionID        string  `json:"vae_version_id"`
	TextEncoderArtifact string  `json:"text_encoder_artifact"`
	TokenizerArtifact   string  `json:"tokenizer_artifact"`
	Stage               Stage   `json:"stage"`
	AuditFrac           float64 `json:"audit_frac"` // 0 = default 5%; negative = none
	// ReqRAMMB is the RAM an encode job asks of its worker (default 8192, the size for full-size configs). Small
	// configs need far less; a laptop with 8 GB total can only take encode jobs when this is lowered.
	ReqRAMMB int `json:"req_ram_mb"`
}

const defaultEncodeRAMMB = 8192

// auditSelected deterministically picks the shards that are re-encoded for audit.
func auditSelected(target string, idx int, frac float64) bool {
	if frac <= 0 {
		return false
	}
	h := sha256.Sum256([]byte(fmt.Sprintf("audit:%s:%d", target, idx)))
	v := float64(uint32(h[0])<<24|uint32(h[1])<<16|uint32(h[2])<<8|uint32(h[3])) / float64(math.MaxUint32)
	return v < frac
}

// StartEncode creates the target latents dataset and one ai.hd_encode job per
// raw shard. Calling it again with the same inputs returns the same dataset.
func (s *Service) StartEncode(ctx context.Context, actor core.Actor, projectID, rawID string, req encodeRequest) (*datasetRow, bool, error) {
	raw, err := s.getDataset(ctx, projectID, rawID)
	if err != nil {
		return nil, false, err
	}
	if raw.Kind != "raw" || raw.Status != "ready" {
		return nil, false, badReq("only a ready raw dataset can be encoded")
	}
	st := req.Stage
	if st.H < 8 || st.W < 8 || st.Frames < 1 || st.FPS < 1 || st.H > 4320 || st.W > 7680 || st.Frames > 2000 || st.FPS > 120 {
		return nil, false, badReq("stage {h,w,frames,fps} out of range")
	}
	if req.ReqRAMMB == 0 {
		req.ReqRAMMB = defaultEncodeRAMMB
	}
	if req.ReqRAMMB < 256 || req.ReqRAMMB > 1<<20 {
		return nil, false, badReq("req_ram_mb must be 256..1048576")
	}
	vaeW, vaeM, err := s.resolveVAE(ctx, projectID, req.VAEVersionID)
	if err != nil {
		return nil, false, err
	}
	te, err1 := s.store.Get(ctx, projectID, req.TextEncoderArtifact)
	tok, err2 := s.store.Get(ctx, projectID, req.TokenizerArtifact)
	if err1 != nil || err2 != nil || te.State != "complete" || tok.State != "complete" {
		return nil, false, badReq("text_encoder_artifact and tokenizer_artifact must be complete artifacts of this project")
	}
	teID := te.SHA256 // the text encoder is identified by its content hash
	shards, err := s.datasetShards(ctx, s.d.DB, raw.ID)
	if err != nil {
		return nil, false, err
	}
	frac := req.AuditFrac
	if frac == 0 {
		frac = defaultAuditFrac
	}
	var trustedDevices int
	_ = s.d.DB.QueryRow(ctx, `SELECT count(*) FROM devices WHERE project_id=$1 AND trusted AND status='active'`, projectID).Scan(&trustedDevices)
	auditNote := ""
	if frac > 0 && trustedDevices < 2 {
		frac, auditNote = 0, "audit re-encode skipped: fewer than 2 trusted devices"
	}
	stj, _ := json.Marshal(st)
	fp := sha256.Sum256([]byte(strings.Join([]string{"encode", raw.Fingerprint, req.VAEVersionID, teID, tok.SHA256, string(stj)}, "|")))
	enc := datasetEncoder{VAEVersionID: req.VAEVersionID, TextEncoderID: teID, TextEncoderArtifact: te.ID, TokenizerArtifact: tok.ID, Stage: &st}
	encJSON, _ := json.Marshal(enc)
	var by any
	if actor.Type == "user" {
		by = actor.ID
	}
	var targetID string
	created := false
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		err := tx.QueryRow(ctx, `INSERT INTO datasets (project_id, name, kind, parent_id, fingerprint, encoder, redistribution, status, detail, created_by)
			VALUES ($1,$2,'latents',$3,$4,$5,$6,'encoding',$7,$8) ON CONFLICT (project_id, kind, fingerprint) DO NOTHING RETURNING id::text`,
			projectID, truncate(raw.Name+"-latents", 80), raw.ID, hex.EncodeToString(fp[:]), encJSON, raw.Redistribution, auditNote, by).Scan(&targetID)
		if errors.Is(err, pgx.ErrNoRows) {
			return tx.QueryRow(ctx, `SELECT id::text FROM datasets WHERE project_id=$1 AND kind='latents' AND fingerprint=$2`, projectID, hex.EncodeToString(fp[:])).Scan(&targetID)
		}
		if err != nil {
			return err
		}
		created = true
		return nil
	})
	if err != nil {
		return nil, false, err
	}
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		for _, sh := range shards {
			for _, kind := range []string{"encode", "audit"} {
				if kind == "audit" && !auditSelected(targetID, sh.Idx, frac) {
					continue
				}
				p := map[string]any{"dataset_id": raw.ID, "target_dataset_id": targetID, "shard_idx": sh.Idx, "shard_sha256": sh.SHA256, "stage": st, "latent": "mean"}
				key := fmt.Sprintf("enc:%s:%s:%d", targetID, req.VAEVersionID, sh.Idx)
				if kind == "audit" {
					p["audit"] = true
					key += ":audit"
				}
				pb, _ := json.Marshal(p)
				id, err := s.sched.Create(ctx, tx, scheduler.NewJob{ProjectID: projectID, JobType: "ai.hd_encode", Params: pb,
					Inputs: []protocol.ArtifactRef{{ID: sh.ArtifactID, Role: "raw_shard"}, {ID: vaeW, Role: "vae_weights"}, {ID: vaeM, Role: "vae_manifest"},
						{ID: te.ID, Role: "text_encoder_weights"}, {ID: tok.ID, Role: "tokenizer"}},
					ReqRAMMB: req.ReqRAMMB, Priority: 15, MaxAttempts: 3, LeaseSeconds: 1800, ReqTrusted: true, IdempotencyKey: key, CreatedBy: by2s(by)})
				if err != nil {
					return err
				}
				if _, err := tx.Exec(ctx, `INSERT INTO dataset_jobs (dataset_id, kind, idx, job_id) VALUES ($1,$2,$3,$4) ON CONFLICT DO NOTHING`, targetID, kind, sh.Idx, id); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err != nil {
		return nil, false, err
	}
	s.d.Events.NotifyJobs(projectID)
	if created {
		s.d.Audit(ctx, nil, actor, "dataset.encode", "dataset:"+targetID, "success", map[string]any{"raw": raw.ID, "vae": req.VAEVersionID, "shards": len(shards)})
	}
	s.Kick()
	d, err := s.getDataset(ctx, projectID, targetID)
	return d, created, err
}

func by2s(v any) *string {
	if s, ok := v.(string); ok {
		return &s
	}
	return nil
}

type dsChild struct {
	kind   string
	idx    int
	jobID  string
	status string
	fail   string
	detail map[string]any
	device string
}

func (s *Service) datasetChildren(ctx context.Context, q core.Querier, id string) ([]dsChild, error) {
	rows, err := q.Query(ctx, `SELECT d.kind, d.idx, d.job_id::text, j.status, j.failure_reason, COALESCE(r.validation_detail, '{}'::jsonb), COALESCE(a.device_id::text, '')
		FROM dataset_jobs d JOIN jobs j ON j.id=d.job_id
		LEFT JOIN job_results r ON r.job_id=j.id AND r.validation_status='accepted'
		LEFT JOIN job_attempts a ON a.id=r.attempt_id
		WHERE d.dataset_id=$1 ORDER BY d.kind, d.idx`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []dsChild
	for rows.Next() {
		var c dsChild
		var det []byte
		if err := rows.Scan(&c.kind, &c.idx, &c.jobID, &c.status, &c.fail, &det, &c.device); err != nil {
			return nil, err
		}
		_ = json.Unmarshal(det, &c.detail)
		out = append(out, c)
	}
	return out, rows.Err()
}

// auditMatches is a necessary condition for latent MSE < 1e-3*var: the means
// and standard deviations of two encodes cannot differ by more than
// sqrt(1e-3)*std. (The per-element MSE needs the tensors; see HD_TRAINING.md.)
func auditMatches(a, b map[string]any) bool {
	am, bm, as, bs := num(a["latent_mean"]), num(b["latent_mean"]), num(a["latent_std"]), num(b["latent_std"])
	tol := 0.0317 * math.Max(as, bs)
	return as > 0 && bs > 0 && math.Abs(am-bm) <= tol && math.Abs(as-bs) <= tol
}

// advanceDataset moves an encoding dataset forward; idempotent.
func (s *Service) advanceDataset(ctx context.Context, id string) error {
	var cancel []string
	var projectID string
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var status string
		var parent *string
		if err := tx.QueryRow(ctx, `SELECT project_id::text, status, parent_id::text FROM datasets WHERE id=$1 FOR UPDATE`, id).Scan(&projectID, &status, &parent); err != nil {
			return err
		}
		if status != "encoding" || parent == nil {
			return nil
		}
		kids, err := s.datasetChildren(ctx, tx, id)
		if err != nil {
			return err
		}
		fail := func(why string) error {
			for _, k := range kids {
				if !scheduler.IsTerminalJob(k.status) {
					cancel = append(cancel, k.jobID)
				}
			}
			_, err := tx.Exec(ctx, `UPDATE datasets SET status='failed', detail=$2, updated_at=now() WHERE id=$1`, id, truncate(why, 900))
			return err
		}
		primary := map[int]dsChild{}
		audits := map[int]dsChild{}
		for _, k := range kids {
			if k.kind == "encode" {
				primary[k.idx] = k
				if k.status == scheduler.JobFailed || k.status == scheduler.JobCancelled {
					return fail(fmt.Sprintf("encode of shard %d %s: %s", k.idx, k.status, k.fail))
				}
			} else {
				audits[k.idx] = k
			}
		}
		raw, err := s.datasetShards(ctx, tx, *parent)
		if err != nil {
			return err
		}
		for _, sh := range raw {
			if p, ok := primary[sh.Idx]; !ok || p.status != scheduler.JobCompleted {
				return nil
			}
		}
		for _, a := range audits {
			if !scheduler.IsTerminalJob(a.status) {
				return nil // audits still running
			}
		}
		var out []datasetShard
		var clips int64
		notes := []string{}
		for _, sh := range raw {
			p := primary[sh.Idx]
			if a, ok := audits[sh.Idx]; ok {
				if a.status == scheduler.JobCompleted {
					if !auditMatches(p.detail, a.detail) {
						return fail(fmt.Sprintf("audit re-encode of shard %d disagrees with the first encode; the encoding device is not trustworthy", sh.Idx))
					}
				} else {
					notes = append(notes, fmt.Sprintf("audit of shard %d inconclusive (%s)", sh.Idx, a.status))
				}
			}
			art, err := resultArtifact(ctx, tx, p.jobID, "latent_shard")
			if err != nil {
				return fail(err.Error())
			}
			var sum string
			var size int64
			if err := tx.QueryRow(ctx, `SELECT sha256, size_bytes FROM artifacts WHERE id=$1`, art).Scan(&sum, &size); err != nil {
				return err
			}
			n := int(num(p.detail["clips"]))
			if n < 1 {
				return fail(fmt.Sprintf("encode of shard %d reported no clips", sh.Idx))
			}
			out = append(out, datasetShard{Idx: sh.Idx, Split: sh.Split, ArtifactID: art, SHA256: sum, Clips: n, Bytes: size})
			clips += int64(n)
		}
		for _, x := range out {
			if _, err := tx.Exec(ctx, `INSERT INTO dataset_shards (dataset_id, idx, split, artifact_id, sha256, clips, bytes) VALUES ($1,$2,$3,$4,$5,$6,$7)
				ON CONFLICT DO NOTHING`, id, x.Idx, x.Split, x.ArtifactID, x.SHA256, x.Clips, x.Bytes); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `UPDATE artifacts SET protected=true WHERE id=$1`, x.ArtifactID); err != nil {
				return err
			}
		}
		_, err = tx.Exec(ctx, `UPDATE datasets SET status='ready', clip_count=$2, detail=$3, updated_at=now() WHERE id=$1`, id, clips, truncate(strings.Join(notes, "; "), 900))
		return err
	})
	if err != nil {
		return err
	}
	for _, j := range cancel {
		if err := s.sched.Cancel(ctx, core.Actor{Type: "system", ID: "dataset"}, projectID, j); err != nil &&
			!errors.Is(err, scheduler.ErrBadState) && !errors.Is(err, scheduler.ErrNotFound) {
			return err
		}
	}
	return nil
}

// tickDatasets advances every encoding dataset (crash recovery included).
func (s *Service) tickDatasets(ctx context.Context) {
	rows, err := s.d.DB.Query(ctx, `SELECT id::text FROM datasets WHERE status='encoding' ORDER BY created_at LIMIT 200`)
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
		_ = s.advanceDataset(ctx, id)
	}
}

// ---- HTTP ----

func (s *Service) datasetView(ctx context.Context, d *datasetRow) (map[string]any, error) {
	shards, err := s.datasetShards(ctx, s.d.DB, d.ID)
	if err != nil {
		return nil, err
	}
	var train, val int
	var bytes int64
	for _, x := range shards {
		if x.Split == "val" {
			val++
		} else {
			train++
		}
		bytes += x.Bytes
	}
	v := map[string]any{"dataset": d, "train_shards": train, "val_shards": val, "bytes": bytes}
	if d.Status == "encoding" {
		kids, err := s.datasetChildren(ctx, s.d.DB, d.ID)
		if err != nil {
			return nil, err
		}
		done, total := 0, 0
		for _, k := range kids {
			if k.kind == "encode" {
				total++
				if k.status == scheduler.JobCompleted {
					done++
				}
			}
		}
		v["encode_done"], v["encode_total"] = done, total
	}
	return v, nil
}

func (s *Service) registerDatasetRoutes(r *api.Router) {
	r.Admin("GET", "/datasets", "viewer", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		rows, err := s.d.DB.Query(req.Context(), `SELECT `+datasetCols+` FROM datasets WHERE project_id=$1 ORDER BY created_at DESC LIMIT 200`, a.ProjectID)
		if err != nil {
			r.Internal(w, req, err)
			return
		}
		var list []*datasetRow
		for rows.Next() {
			if d, err := scanDataset(rows); err == nil {
				list = append(list, d)
			}
		}
		rows.Close()
		out := []map[string]any{}
		for _, d := range list {
			v, err := s.datasetView(req.Context(), d)
			if err != nil {
				r.Internal(w, req, err)
				return
			}
			out = append(out, v)
		}
		api.WriteJSON(w, 200, map[string]any{"datasets": out})
	})
	r.Admin("GET", "/datasets/{did}", "viewer", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		d, err := s.getDataset(req.Context(), a.ProjectID, req.PathValue("did"))
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		v, err := s.datasetView(req.Context(), d)
		if err != nil {
			r.Internal(w, req, err)
			return
		}
		api.WriteJSON(w, 200, v)
	})
	r.Admin("POST", "/datasets", "operator", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		var body registerRequest
		if err := api.DecodeJSON(w, req, 8<<20, &body); err != nil {
			api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request: "+err.Error())
			return
		}
		d, err := s.RegisterDataset(req.Context(), a.Actor, a.ProjectID, body)
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		api.WriteJSON(w, 201, d)
	})
	r.Admin("POST", "/datasets/{did}/encode", "operator", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		var body encodeRequest
		if err := api.DecodeJSON(w, req, 16<<10, &body); err != nil {
			api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request: "+err.Error())
			return
		}
		d, created, err := s.StartEncode(req.Context(), a.Actor, a.ProjectID, req.PathValue("did"), body)
		if err != nil {
			s.writeErr(w, req, r, err)
			return
		}
		code := 200
		if created {
			code = 201
		}
		api.WriteJSON(w, code, map[string]any{"dataset": d, "created": created})
	})
	s.registerUploadRoutes(r)
}

// ---- resumable operator uploads (8 MB chunks), same store as the worker API ----

func uploadStatus(i artifacts.Info) protocol.UploadStatus {
	st := protocol.UploadStatus{UploadID: i.ID, Offset: i.Received, Size: i.Size, ChunkSize: artifacts.ChunkSize, State: i.State}
	if i.State == "complete" {
		st.ArtifactID = i.ID
		st.Offset = i.Size
	}
	return st
}

func (s *Service) artifactErr(w http.ResponseWriter, req *http.Request, r *api.Router, err error, info artifacts.Info) {
	switch {
	case errors.Is(err, artifacts.ErrOffsetMismatch):
		api.WriteJSON(w, 409, map[string]any{"error": protocol.ErrorDetail{Code: protocol.ErrOffsetMismatch, Message: err.Error()}, "status": uploadStatus(info)})
	case errors.Is(err, artifacts.ErrNotFound):
		api.WriteErr(w, 404, protocol.ErrNotFound, err.Error())
	case errors.Is(err, artifacts.ErrForbidden):
		api.WriteErr(w, 403, protocol.ErrForbidden, err.Error())
	case errors.Is(err, artifacts.ErrTooLarge):
		api.WriteErr(w, 413, protocol.ErrTooLarge, err.Error())
	case errors.Is(err, artifacts.ErrNoSpace):
		api.WriteErr(w, 507, protocol.ErrUnavailable, err.Error())
	case errors.Is(err, artifacts.ErrChecksum):
		api.WriteErr(w, 422, protocol.ErrChecksum, err.Error())
	case errors.Is(err, artifacts.ErrIncomplete), errors.Is(err, artifacts.ErrBadRequest):
		api.WriteErr(w, 400, protocol.ErrBadRequest, err.Error())
	default:
		r.Internal(w, req, err)
	}
}

// ownUpload loads an upload that an operator (not a worker) started.
func (s *Service) ownUpload(ctx context.Context, projectID, id string) (artifacts.Info, error) {
	if !uuidRe.MatchString(id) {
		return artifacts.Info{}, artifacts.ErrNotFound
	}
	info, err := s.store.Get(ctx, projectID, id)
	if err != nil {
		return info, err
	}
	if info.UploadedBy != nil {
		return info, artifacts.ErrForbidden
	}
	return info, nil
}

func (s *Service) registerUploadRoutes(r *api.Router) {
	r.Admin("POST", "/artifacts/uploads", "operator", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		var body struct {
			Kind   string `json:"kind"`
			SHA256 string `json:"sha256"`
			Size   int64  `json:"size"`
		}
		if err := api.DecodeJSON(w, req, 4096, &body); err != nil {
			api.WriteErr(w, 400, protocol.ErrBadRequest, "malformed request")
			return
		}
		body.SHA256 = strings.ToLower(body.SHA256)
		// Beginning the same upload twice resumes the unfinished one.
		var existing string
		err := s.d.DB.QueryRow(req.Context(), `SELECT id::text FROM artifacts WHERE project_id=$1 AND kind=$2 AND sha256=$3 AND size_bytes=$4
			AND state='uploading' AND uploaded_by_user=$5 AND uploaded_by_device IS NULL ORDER BY created_at DESC LIMIT 1`,
			a.ProjectID, body.Kind, body.SHA256, body.Size, a.Actor.ID).Scan(&existing)
		if err == nil {
			if info, err := s.store.Get(req.Context(), a.ProjectID, existing); err == nil {
				api.WriteJSON(w, 200, uploadStatus(info))
				return
			}
		}
		info, err := s.store.Begin(req.Context(), a.ProjectID, artifacts.Owner{UserID: a.Actor.ID}, body.Kind, body.SHA256, "", body.Size)
		if err != nil {
			s.artifactErr(w, req, r, err, info)
			return
		}
		api.WriteJSON(w, 200, uploadStatus(info))
	})
	r.Admin("GET", "/artifacts/uploads/{id}", "operator", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		info, err := s.ownUpload(req.Context(), a.ProjectID, req.PathValue("id"))
		if err != nil {
			s.artifactErr(w, req, r, err, info)
			return
		}
		api.WriteJSON(w, 200, uploadStatus(info))
	})
	r.Admin("PUT", "/artifacts/uploads/{id}", "operator", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		off, err := strconv.ParseInt(req.URL.Query().Get("offset"), 10, 64)
		if err != nil || off < 0 {
			api.WriteErr(w, 400, protocol.ErrBadRequest, "offset query parameter required")
			return
		}
		if info, err := s.ownUpload(req.Context(), a.ProjectID, req.PathValue("id")); err != nil {
			s.artifactErr(w, req, r, err, info)
			return
		}
		info, err := s.store.WriteChunk(req.Context(), a.ProjectID, req.PathValue("id"), artifacts.Owner{UserID: a.Actor.ID}, off, req.Body)
		if err != nil {
			s.artifactErr(w, req, r, err, info)
			return
		}
		api.WriteJSON(w, 200, uploadStatus(info))
	})
	r.Admin("POST", "/artifacts/uploads/{id}/complete", "operator", func(w http.ResponseWriter, req *http.Request, a *api.AdminCtx) {
		if info, err := s.ownUpload(req.Context(), a.ProjectID, req.PathValue("id")); err != nil {
			s.artifactErr(w, req, r, err, info)
			return
		}
		info, err := s.store.Finish(req.Context(), a.ProjectID, req.PathValue("id"), artifacts.Owner{UserID: a.Actor.ID})
		if err != nil {
			s.artifactErr(w, req, r, err, info)
			return
		}
		s.d.Audit(req.Context(), nil, a.Actor, "artifact.upload", "artifact:"+info.ID, "success", map[string]any{"kind": info.Kind, "bytes": info.Size})
		api.WriteJSON(w, 200, uploadStatus(info))
	})
}
