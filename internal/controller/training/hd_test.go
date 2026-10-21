package training

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"nivra/internal/controller/artifacts"
	"nivra/internal/controller/core"
	"nivra/internal/controller/projects"
	"nivra/internal/controller/scheduler"
	"nivra/internal/shared/protocol"
	"nivra/internal/testutil"
)

// ---- pure tests ----

func TestSplitmix64KnownVector(t *testing.T) {
	st := uint64(0)
	if got := splitmix64(&st); got != 0xe220a8397b1dcdaf {
		t.Fatalf("splitmix64(0)=%#x", got)
	}
}

// Golden values were produced by an independent implementation of the
// documented algorithm (Fisher-Yates from the end, splitmix64 seeded with
// DataSeed xor epoch; global position slot*k+j, epoch = position / S).
func TestHDShardAssignmentGolden(t *testing.T) {
	if got, want := hdPermutation(10, 42, 0), []int{0, 9, 5, 8, 6, 4, 7, 2, 1, 3}; !reflect.DeepEqual(got, want) {
		t.Fatalf("perm epoch0 %v", got)
	}
	if got, want := hdPermutation(10, 42, 1), []int{4, 2, 5, 6, 1, 3, 9, 8, 7, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("perm epoch1 %v", got)
	}
	const W = 4
	for _, c := range []struct {
		round, batch, i int
		slot            int
		want            []int
	}{
		{0, 0, 0, 0, []int{0, 9, 5}},
		{0, 0, 3, 3, []int{3, 4, 2}}, // straddles the epoch boundary
		{0, 1, 0, 4, []int{5, 6, 1}},
		{1, 0, 0, 32, []int{0, 7, 1}},
		{2, 5, 3, 87, []int{3, 5, 0}},
	} {
		slot := hdSlot(c.round, c.batch, W, c.i)
		if slot != c.slot {
			t.Fatalf("slot(%d,%d,%d)=%d want %d", c.round, c.batch, c.i, slot, c.slot)
		}
		if got := hdAssign(10, 3, slot, 42); !reflect.DeepEqual(got, c.want) {
			t.Fatalf("assign slot %d = %v want %v", slot, got, c.want)
		}
	}
	// Within one epoch every shard is used exactly once.
	seen := map[int]int{}
	for slot := 0; slot < 5; slot++ {
		for _, x := range hdAssign(10, 2, slot, 7) {
			seen[x]++
		}
	}
	if len(seen) != 10 {
		t.Fatalf("epoch did not cover all shards: %v", seen)
	}
	if hdUpdateID("r", 1, 2, 3) != hdUpdateID("r", 1, 2, 3) || hdUpdateID("r", 1, 2, 3) == hdUpdateID("r", 1, 2, 4) {
		t.Fatal("update ids must be deterministic and distinct")
	}
	if !uuidRe.MatchString(hdUpdateID("r", 0, 0, 0)) {
		t.Fatal("update id is not uuid-shaped")
	}
}

func TestSmallRunConfigJSONHasNoHDFields(t *testing.T) {
	c := RunConfig{}
	c.Defaults()
	b, _ := json.Marshal(c)
	for _, k := range []string{"family", "target", "hd_config", "dataset_id", "stage", "req_vram_mb", "eval_gate", "retention", "warmup_steps", "grad_accum"} {
		if strings.Contains(string(b), `"`+k+`"`) {
			t.Fatalf("small config JSON leaks %q: %s", k, b)
		}
	}
	if c.WorkersPerRound != 2 || c.Inner.InnerSteps != 30 || c.LeaseSeconds != 300 {
		t.Fatalf("small defaults changed: %+v", c)
	}
}

func TestHDDefaultsAndValidation(t *testing.T) {
	id := "11111111-1111-4111-8111-111111111111"
	c := RunConfig{HDFields: HDFields{Family: "hd", HDConfig: "nivra_base", DatasetID: id, InitVersionID: id}}
	c.Defaults()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if c.WorkersPerRound != 4 || c.MinUpdates != 2 || c.RoundTimeoutSec != 7200 || c.Inner.InnerSteps != 500 || c.Inner.BatchSize != 4 ||
		c.Inner.GradAccum != 2 || c.Inner.LR != 1e-4 || c.OuterLR != 0.7 || c.Momentum != 0.9 || c.ReqVRAMMB != 5500 || c.LeaseSeconds != 900 ||
		c.EvalGate != "pi" || c.BoundC != 3 || c.Retention.KeepLast != 5 || c.Retention.KeepEvery != 10 || c.Inner.WarmupSteps != 20 {
		t.Fatalf("hd defaults: %+v", c)
	}
	m := RunConfig{HDFields: HDFields{Family: "hd", HDConfig: "nivra_medium", DatasetID: id, InitVersionID: id}}
	m.Defaults()
	if m.ReqVRAMMB != 22000 || m.EvalGate != "worker" {
		t.Fatalf("medium defaults: %+v", m)
	}
	bad := c
	bad.RedundancyFrac = 0.1
	if bad.Validate() == nil {
		t.Fatal("redundancy_frac must be rejected until implemented")
	}
	bad = c
	bad.Family = "weird"
	if bad.Validate() == nil {
		t.Fatal("unknown family accepted")
	}
}

// ---- safetensors fixtures ----

func f16bits(v float32) uint16 {
	b := math.Float32bits(v)
	sign := uint16(b>>16) & 0x8000
	exp := int((b>>23)&0xff) - 127
	frac := b & 0x7fffff
	switch {
	case v == 0:
		return sign
	case exp > 15:
		return sign | 0x7c00
	case exp < -14:
		return sign | uint16(math.Round(math.Abs(float64(v))/math.Ldexp(1, -24)))
	}
	return sign | uint16(exp+15)<<10 | uint16(frac>>13)
}

type stFix struct {
	name  string
	dtype string
	shape []int64
	data  []byte
}

func f16data(vals ...float32) []byte {
	out := make([]byte, 2*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint16(out[2*i:], f16bits(v))
	}
	return out
}

func fill16(n int, v float32) []byte {
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = v
	}
	return f16data(vals...)
}

func stBytes(ts []stFix, extra int) []byte {
	sort.Slice(ts, func(a, b int) bool { return ts[a].name < ts[b].name })
	hdr := map[string]any{}
	var pos int64
	var data []byte
	for _, t := range ts {
		hdr[t.name] = map[string]any{"dtype": t.dtype, "shape": t.shape, "data_offsets": []int64{pos, pos + int64(len(t.data))}}
		pos += int64(len(t.data))
		data = append(data, t.data...)
	}
	hj, _ := json.Marshal(hdr)
	out := make([]byte, 8, 8+len(hj)+len(data)+extra)
	binary.LittleEndian.PutUint64(out, uint64(len(hj)))
	out = append(out, hj...)
	out = append(out, data...)
	return append(out, make([]byte, extra)...)
}

func writeTemp(t *testing.T, b []byte) string {
	p := filepath.Join(t.TempDir(), "x.safetensors")
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

var fixBase = func() []stFix {
	return []stFix{{"a", "F16", []int64{4, 4}, fill16(16, 0)}, {"b", "F16", []int64{8}, fill16(8, 0)}}
}

func TestDeltaStructureAndScan(t *testing.T) {
	bh, err := readSafetensorsHeader(writeTemp(t, stBytes(fixBase(), 0)))
	if err != nil {
		t.Fatal(err)
	}
	good := []stFix{{"a", "F16", []int64{4, 4}, fill16(16, 0.001)}, {"b", "F16", []int64{8}, fill16(8, -0.001)}}
	gp := writeTemp(t, stBytes(good, 0))
	gh, err := readSafetensorsHeader(gp)
	if err != nil {
		t.Fatal(err)
	}
	if why := checkDeltaStructure(gh, bh); why != "" {
		t.Fatal(why)
	}
	if want := int64(8+len(mustJSONLen(gp))) + (16+8)*2; gh.Size != want {
		t.Logf("size %d", gh.Size)
	}
	st, why, err := scanDelta(gp, gh, 0.01)
	if err != nil || why != "" || !st.Nonzero || st.Linf <= 0 {
		t.Fatalf("scan good: %+v %q %v", st, why, err)
	}
	// Bound exceeded.
	if _, why, _ := scanDelta(gp, gh, 0.0005); !strings.Contains(why, "exceeds") {
		t.Fatalf("bound not enforced: %q", why)
	}
	// All zeros.
	zp := writeTemp(t, stBytes(fixBase(), 0))
	zh, _ := readSafetensorsHeader(zp)
	if _, why, _ := scanDelta(zp, zh, 1); !strings.Contains(why, "zeros") {
		t.Fatalf("zero update accepted: %q", why)
	}
	// NaN / Inf.
	nan := []stFix{{"a", "F16", []int64{4, 4}, append(fill16(15, 0.001), 0x00, 0x7e)}, {"b", "F16", []int64{8}, fill16(8, 0.001)}}
	np := writeTemp(t, stBytes(nan, 0))
	nh, _ := readSafetensorsHeader(np)
	if _, why, _ := scanDelta(np, nh, 1); !strings.Contains(why, "NaN") {
		t.Fatalf("NaN accepted: %q", why)
	}
	// Trailing bytes break the tiling (exact size).
	if _, err := readSafetensorsHeader(writeTemp(t, stBytes(good, 4))); err == nil {
		t.Fatal("padded file accepted")
	}
	// Wrong dtype, shape, missing and extra tensors.
	for name, fix := range map[string][]stFix{
		"f32":     {{"a", "F32", []int64{4, 4}, make([]byte, 64)}, {"b", "F16", []int64{8}, fill16(8, 0.001)}},
		"shape":   {{"a", "F16", []int64{2, 8}, fill16(16, 0.001)}, {"b", "F16", []int64{8}, fill16(8, 0.001)}},
		"missing": {{"a", "F16", []int64{4, 4}, fill16(16, 0.001)}},
		"extra":   append(append([]stFix{}, good...), stFix{"c", "F16", []int64{1}, fill16(1, 0.1)}),
		"renamed": {{"a", "F16", []int64{4, 4}, fill16(16, 0.001)}, {"z", "F16", []int64{8}, fill16(8, 0.001)}},
	} {
		h, err := readSafetensorsHeader(writeTemp(t, stBytes(fix, 0)))
		if err != nil {
			t.Fatal(name, err)
		}
		if checkDeltaStructure(h, bh) == "" {
			t.Fatalf("%s accepted", name)
		}
	}
}

func mustJSONLen(p string) []byte { b, _ := os.ReadFile(p); return b[:8] }

func TestHDEchoMismatch(t *testing.T) {
	p := hdTrainParams{UpdateID: "u", RunID: "r", Round: 2, RoundBatch: 1, BaseVersionID: "v", Target: "dit", HDConfigSHA: "sha",
		Stage: Stage{256, 256, 17, 8}, InnerSteps: 10, BatchSize: 2, GradAccum: 2, LR: 1e-4, DeltaDtype: "float16",
		Shards: []shardRef{{"a1", "s1"}, {"a2", "s2"}}}
	good := func() hdUpdateManifest {
		return hdUpdateManifest{Schema: "nivra.update/2", RunID: "r", Round: 2, RoundBatch: 1, BaseVersionID: "v", UpdateID: "u", AttemptID: "att",
			Target: "dit", HDConfigSHA: "sha", Stage: p.Stage, InnerSteps: 10, LR: 1e-4, Shards: append([]shardRef{}, p.Shards...), SamplesSeen: 40, DeltaDtype: "float16"}
	}
	if why := hdEchoMismatch(p, good(), "att"); why != "" {
		t.Fatal(why)
	}
	muts := map[string]func(*hdUpdateManifest){
		"run_id": func(m *hdUpdateManifest) { m.RunID = "x" }, "round": func(m *hdUpdateManifest) { m.Round = 3 },
		"round_batch": func(m *hdUpdateManifest) { m.RoundBatch = 0 }, "base_version_id": func(m *hdUpdateManifest) { m.BaseVersionID = "w" },
		"update_id": func(m *hdUpdateManifest) { m.UpdateID = "x" }, "target": func(m *hdUpdateManifest) { m.Target = "vae" },
		"hd_config_sha": func(m *hdUpdateManifest) { m.HDConfigSHA = "x" }, "stage": func(m *hdUpdateManifest) { m.Stage.H = 128 },
		"inner_steps": func(m *hdUpdateManifest) { m.InnerSteps = 11 }, "lr": func(m *hdUpdateManifest) { m.LR = 2e-4 },
		"delta_dtype": func(m *hdUpdateManifest) { m.DeltaDtype = "float32" }, "shards": func(m *hdUpdateManifest) { m.Shards[1].SHA256 = "zz" },
		"shards-len": func(m *hdUpdateManifest) { m.Shards = m.Shards[:1] }, "samples": func(m *hdUpdateManifest) { m.SamplesSeen = 41 },
		"schema": func(m *hdUpdateManifest) { m.Schema = "foo" },
	}
	for name, mut := range muts {
		m := good()
		mut(&m)
		if hdEchoMismatch(p, m, "att") == "" {
			t.Fatalf("%s mismatch not detected", name)
		}
	}
	if hdEchoMismatch(p, good(), "other-attempt") == "" {
		t.Fatal("attempt_id mismatch not detected")
	}
}

func TestAuditMatches(t *testing.T) {
	a := map[string]any{"latent_mean": 0.1, "latent_std": 1.0}
	if !auditMatches(a, map[string]any{"latent_mean": 0.11, "latent_std": 1.01}) || auditMatches(a, map[string]any{"latent_mean": 0.3, "latent_std": 1.0}) {
		t.Fatal("audit tolerance")
	}
}

// ---- database tests ----

type hdEnv struct {
	t       *testing.T
	s       *Service
	ctx     context.Context
	project string
	other   string
}

func newHDEnv(t *testing.T) *hdEnv {
	t.Helper()
	d := testutil.Deps(t)
	reg := scheduler.NewRegistry()
	sched := scheduler.New(d, reg)
	store, err := artifacts.New(d)
	if err != nil {
		t.Fatal(err)
	}
	svc := &Service{d: d, sched: sched, store: store, kick: make(chan struct{}, 1)}
	RegisterJobTypes(reg, svc)
	ctx := context.Background()
	ps := projects.New(d)
	p1, _, err := ps.Create(ctx, core.System, "HD one")
	if err != nil {
		t.Fatal(err)
	}
	p2, _, err := ps.Create(ctx, core.System, "HD two")
	if err != nil {
		t.Fatal(err)
	}
	return &hdEnv{t: t, s: svc, ctx: ctx, project: p1.ID, other: p2.ID}
}

func (e *hdEnv) art(project, kind string, data []byte) artifacts.Info {
	e.t.Helper()
	i, err := e.s.store.ImportBytes(e.ctx, project, kind, data, "")
	if err != nil {
		e.t.Fatal(err)
	}
	return i
}

func randBytes(n int) []byte { b := make([]byte, n); _, _ = rand.Read(b); return b }

func (e *hdEnv) shards(project, artKind string, n int, valFrom int) []datasetShard {
	var out []datasetShard
	for i := 0; i < n; i++ {
		a := e.art(project, artKind, randBytes(64))
		split := "train"
		if i >= valFrom {
			split = "val"
		}
		out = append(out, datasetShard{Idx: i, Split: split, ArtifactID: a.ID, SHA256: a.SHA256, Clips: 3, Bytes: a.Size})
	}
	return out
}

func TestMigration0008(t *testing.T) {
	e := newHDEnv(t)
	for _, q := range []string{
		`SELECT count(*) FROM datasets`, `SELECT count(*) FROM dataset_shards`, `SELECT count(*) FROM dataset_jobs`, `SELECT count(*) FROM hd_candidates`,
		`SELECT train_weights_artifact, family, compat_sha FROM model_versions`, `SELECT content_sha256 FROM training_updates`,
	} {
		rows, err := e.s.d.DB.Query(e.ctx, q)
		if err != nil {
			t.Fatalf("%s: %v", q, err)
		}
		rows.Close()
	}
	var n int
	if err := e.s.d.DB.QueryRow(e.ctx, `SELECT count(*) FROM pg_indexes WHERE indexname IN ('training_updates_content_uq','datasets_fingerprint_uq')`).Scan(&n); err != nil || n != 2 {
		t.Fatalf("indexes: %d %v", n, err)
	}
}

func TestRegisterDataset(t *testing.T) {
	e := newHDEnv(t)
	sh := e.shards(e.project, "dataset_shard", 3, 2)
	req := registerRequest{Name: "clips", Kind: "raw", Shards: sh}
	d, err := e.s.RegisterDataset(e.ctx, core.System, e.project, req)
	if err != nil {
		t.Fatal(err)
	}
	if d.ClipCount != 9 || d.Status != "ready" || d.Redistribution != "owner_devices" || d.Fingerprint == "" {
		t.Fatalf("%+v", d)
	}
	// Same content again: conflict.
	if _, err := e.s.RegisterDataset(e.ctx, core.System, e.project, req); !errors.Is(err, ErrBadState) {
		t.Fatalf("duplicate registration: %v", err)
	}
	fresh := func() registerRequest {
		return registerRequest{Name: "n", Kind: "raw", Shards: e.shards(e.project, "dataset_shard", 2, 2)}
	}
	r := fresh()
	r.Shards[1].SHA256 = strings.Repeat("0", 64)
	if _, err := e.s.RegisterDataset(e.ctx, core.System, e.project, r); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("bad sha: %v", err)
	}
	r = fresh()
	r.Shards[0].ArtifactID = e.art(e.other, "dataset_shard", randBytes(32)).ID // other project's artifact
	r.Shards[0].SHA256 = e.art(e.other, "dataset_shard", randBytes(32)).SHA256
	if _, err := e.s.RegisterDataset(e.ctx, core.System, e.project, r); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("wrong project: %v", err)
	}
	r = fresh()
	r.Shards[1].Idx = 0
	if _, err := e.s.RegisterDataset(e.ctx, core.System, e.project, r); err == nil {
		t.Fatal("duplicate idx accepted")
	}
	r = fresh()
	r.Kind = "latents"
	r.Encoder = datasetEncoder{VAEVersionID: "x", TextEncoderID: "t", Stage: &Stage{64, 64, 9, 8}}
	if _, err := e.s.RegisterDataset(e.ctx, core.System, e.project, r); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("raw shard kind in latents dataset: %v", err)
	}
	r = fresh()
	r.Shards[0].Split = "test"
	if _, err := e.s.RegisterDataset(e.ctx, core.System, e.project, r); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("bad split: %v", err)
	}
	// Incomplete upload.
	inc, err := e.s.store.Begin(e.ctx, e.project, artifacts.Owner{UserID: ""}, "dataset_shard", strings.Repeat("a", 64), "", 10)
	if err != nil {
		t.Fatal(err)
	}
	r = fresh()
	r.Shards[0] = datasetShard{Idx: 0, Split: "train", ArtifactID: inc.ID, SHA256: inc.SHA256, Clips: 1}
	if _, err := e.s.RegisterDataset(e.ctx, core.System, e.project, r); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("incomplete artifact: %v", err)
	}
}

// vaeVersion inserts a model version usable as a frozen VAE.
func (e *hdEnv) version(project string, man map[string]any, w, manifest artifacts.Info, vae *string) string {
	e.t.Helper()
	id := newUUID()
	mb, _ := json.Marshal(man)
	if _, err := e.s.d.DB.Exec(e.ctx, `INSERT INTO model_versions (id, project_id, version_number, weights_artifact, manifest_artifact, manifest, vae_artifact)
		VALUES ($1,$2,0,$3,$4,$5,$6)`, id, project, w.ID, manifest.ID, mb, vae); err != nil {
		e.t.Fatal(err)
	}
	return id
}

// completeJob marks a job done with an accepted attempt, optionally with an output artifact.
func (e *hdEnv) completeJob(jobID, device, outKind string, out []byte, detail map[string]any) string {
	e.t.Helper()
	var project string
	if err := e.s.d.DB.QueryRow(e.ctx, `SELECT project_id::text FROM jobs WHERE id=$1`, jobID).Scan(&project); err != nil {
		e.t.Fatal(err)
	}
	var att string
	if err := e.s.d.DB.QueryRow(e.ctx, `INSERT INTO job_attempts (job_id, device_id, attempt_number, fencing_token, state, lease_expires_at)
		VALUES ($1,$2,1,nextval('fencing_seq'),'accepted', now() + interval '1 hour') RETURNING id::text`, jobID, device).Scan(&att); err != nil {
		e.t.Fatal(err)
	}
	artID := ""
	if out != nil {
		a := e.art(project, outKind, out)
		artID = a.ID
		if _, err := e.s.d.DB.Exec(e.ctx, `UPDATE artifacts SET attempt_id=$2 WHERE id=$1`, a.ID, att); err != nil {
			e.t.Fatal(err)
		}
	}
	db, _ := json.Marshal(detail)
	if _, err := e.s.d.DB.Exec(e.ctx, `INSERT INTO job_results (attempt_id, job_id, artifact_id, checksum, validation_status, validation_detail, accepted_at)
		VALUES ($1,$2,NULLIF($3,'')::uuid,'x','accepted',$4, now())`, att, jobID, artID, db); err != nil {
		e.t.Fatal(err)
	}
	if _, err := e.s.d.DB.Exec(e.ctx, `UPDATE jobs SET status='completed', current_attempt_id=$2 WHERE id=$1`, jobID, att); err != nil {
		e.t.Fatal(err)
	}
	return att
}

func (e *hdEnv) device(project, name string) string {
	e.t.Helper()
	var id string
	if err := e.s.d.DB.QueryRow(e.ctx, `INSERT INTO devices (project_id, public_key, name, trusted) VALUES ($1,$2,$3,true) RETURNING id::text`,
		project, randBytes(32), name).Scan(&id); err != nil {
		e.t.Fatal(err)
	}
	return id
}

func TestEncodeIdempotentAndAdvance(t *testing.T) {
	e := newHDEnv(t)
	raw, err := e.s.RegisterDataset(e.ctx, core.System, e.project, registerRequest{Name: "raw", Kind: "raw", Shards: e.shards(e.project, "dataset_shard", 3, 2)})
	if err != nil {
		t.Fatal(err)
	}
	vae := e.version(e.project, map[string]any{"target": "vae", "family": "hd"}, e.art(e.project, "model_weights", randBytes(16)), e.art(e.project, "model_version", []byte(`{}`)), nil)
	te, tok := e.art(e.project, "text_encoder", randBytes(16)), e.art(e.project, "tokenizer", randBytes(16))
	req := encodeRequest{VAEVersionID: vae, TextEncoderArtifact: te.ID, TokenizerArtifact: tok.ID, Stage: Stage{64, 64, 9, 8}}
	d1, created, err := e.s.StartEncode(e.ctx, core.System, e.project, raw.ID, req)
	if err != nil || !created {
		t.Fatalf("encode: %v %v", created, err)
	}
	d2, created2, err := e.s.StartEncode(e.ctx, core.System, e.project, raw.ID, req)
	if err != nil || created2 || d2.ID != d1.ID {
		t.Fatalf("second encode must return the same dataset: %v %v %+v", created2, err, d2)
	}
	var jobs int
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT count(*) FROM jobs WHERE job_type='ai.hd_encode' AND req_trusted`).Scan(&jobs)
	if jobs != 3 {
		t.Fatalf("encode jobs %d", jobs)
	}
	if d1.Encoder.TextEncoderID != te.SHA256 || d1.Status != "encoding" || d1.ParentID == nil {
		t.Fatalf("%+v", d1)
	}
	// Not done yet: advancing is a no-op.
	if err := e.s.advanceDataset(e.ctx, d1.ID); err != nil {
		t.Fatal(err)
	}
	dev := e.device(e.project, "w1")
	rows, _ := e.s.d.DB.Query(e.ctx, `SELECT d.idx, d.job_id::text FROM dataset_jobs d WHERE d.dataset_id=$1 ORDER BY idx`, d1.ID)
	type jr struct {
		idx int
		id  string
	}
	var js []jr
	for rows.Next() {
		var x jr
		_ = rows.Scan(&x.idx, &x.id)
		js = append(js, x)
	}
	rows.Close()
	for _, j := range js {
		payload := randBytes(40)
		e.completeJob(j.id, dev, "latent_shard", payload, map[string]any{"clips": 3, "latent_mean": 0.0, "latent_std": 1.0})
	}
	for i := 0; i < 2; i++ { // twice: idempotent
		if err := e.s.advanceDataset(e.ctx, d1.ID); err != nil {
			t.Fatal(err)
		}
	}
	got, _ := e.s.getDataset(e.ctx, e.project, d1.ID)
	if got.Status != "ready" || got.ClipCount != 9 {
		t.Fatalf("after advance: %+v", got)
	}
	sh, _ := e.s.datasetShards(e.ctx, e.s.d.DB, d1.ID)
	if len(sh) != 3 || sh[2].Split != "val" || sh[0].Split != "train" {
		t.Fatalf("shards %+v", sh)
	}
}

// hdRunFixture builds a latents dataset, an init version and a running HD run.
type hdRunFixture struct {
	e       *hdEnv
	run     *Run
	dataset *datasetRow
	base    []byte
	vae     string
}

func newRunFixture(t *testing.T, workers int, gate string) *hdRunFixture {
	e := newHDEnv(t)
	vae := e.version(e.project, map[string]any{"target": "vae", "family": "hd"}, e.art(e.project, "model_weights", randBytes(16)), e.art(e.project, "model_version", []byte(`{}`)), nil)
	ds, err := e.s.RegisterDataset(e.ctx, core.System, e.project, registerRequest{Name: "lat", Kind: "latents",
		Encoder: datasetEncoder{VAEVersionID: vae, TextEncoderID: "te-1", Stage: &Stage{64, 64, 9, 8}},
		Shards:  e.shards(e.project, "latent_shard", 6, 5)})
	if err != nil {
		t.Fatal(err)
	}
	base := stBytes(fixBase(), 0)
	w := e.art(e.project, "model_weights", base)
	m := e.art(e.project, "model_version", []byte(`{}`))
	init := e.version(e.project, map[string]any{"family": "hd", "target": "dit", "hd_config_sha": "cfgsha", "version_id": "v-init"}, w, m, nil)
	cfg := RunConfig{WorkersPerRound: workers, MinUpdates: 1, DataSeed: 42, Rounds: 3,
		HDFields: HDFields{Family: "hd", HDConfig: "nivra_hd_tiny", DatasetID: ds.ID, InitVersionID: init, EvalGate: gate, ShardsPerJob: 2}}
	cfg.Inner.InnerSteps, cfg.Inner.WarmupSteps = 10, 2
	run, err := e.s.CreateRun(e.ctx, core.System, e.project, "hd run", cfg, "")
	if err != nil {
		t.Fatal(err)
	}
	f := &hdRunFixture{e: e, dataset: ds, base: base, vae: vae}
	f.run = f.step()
	if f.run.Status != "running" || f.run.ModelVersion == nil {
		t.Fatalf("run after init: %+v", f.run)
	}
	_ = run
	return f
}

func (f *hdRunFixture) step() *Run {
	f.e.t.Helper()
	var id string
	if err := f.e.s.d.DB.QueryRow(f.e.ctx, `SELECT id::text FROM training_runs WHERE project_id=$1`, f.e.project).Scan(&id); err != nil {
		f.e.t.Fatal(err)
	}
	r, err := f.e.s.runByID(f.e.ctx, id)
	if err != nil {
		f.e.t.Fatal(err)
	}
	if err := f.e.s.step(f.e.ctx, r); err != nil {
		f.e.t.Fatal(err)
	}
	r, _ = f.e.s.runByID(f.e.ctx, id)
	return r
}

type jobRow struct {
	id, key string
	params  hdTrainParams
	roles   []string
	ids     []string
	trusted bool
	vram    int
}

func (f *hdRunFixture) jobs() []jobRow {
	rows, err := f.e.s.d.DB.Query(f.e.ctx, `SELECT id::text, idempotency_key, params, input_roles, input_artifacts::text[], req_trusted, req_vram_mb
		FROM jobs WHERE job_type='ai.hd_train' ORDER BY idempotency_key`)
	if err != nil {
		f.e.t.Fatal(err)
	}
	defer rows.Close()
	var out []jobRow
	for rows.Next() {
		var j jobRow
		var pb []byte
		if err := rows.Scan(&j.id, &j.key, &pb, &j.roles, &j.ids, &j.trusted, &j.vram); err != nil {
			f.e.t.Fatal(err)
		}
		_ = json.Unmarshal(pb, &j.params)
		out = append(out, j)
	}
	return out
}

func TestHDRoundJobsDeterministicAndIdempotent(t *testing.T) {
	f := newRunFixture(t, 2, "off")
	f.step()
	js := f.jobs()
	if len(js) != 2 {
		t.Fatalf("want 2 jobs got %d", len(js))
	}
	f.step() // again: no duplicates
	f.step()
	if got := f.jobs(); len(got) != 2 {
		t.Fatalf("jobs duplicated: %d", len(got))
	}
	train, _ := splitShards(mustShards(t, f))
	if len(train) != 5 {
		t.Fatalf("train %d", len(train))
	}
	for i, j := range js {
		if j.key != fmt.Sprintf("run:%s:r0:b0:w%d", f.run.ID, i) {
			t.Fatalf("key %s", j.key)
		}
		want := hdAssign(len(train), 2, hdSlot(0, 0, 2, i), 42)
		if len(j.params.Shards) != 2 || j.params.Shards[0].ArtifactID != train[want[0]].ArtifactID || j.params.Shards[1].SHA256 != train[want[1]].SHA256 {
			t.Fatalf("job %d shards %+v want idx %v", i, j.params.Shards, want)
		}
		wantRoles := []string{"base_train_weights", "base_manifest", "data_shard_000", "data_shard_001"}
		if !reflect.DeepEqual(j.roles, wantRoles) {
			t.Fatalf("roles %v", j.roles)
		}
		p := j.params
		if p.UpdateID != hdUpdateID(f.run.ID, 0, 0, i) || p.RunID != f.run.ID || p.Round != 0 || p.RoundBatch != 0 || p.Target != "dit" ||
			p.HDConfig != "nivra_hd_tiny" || p.HDConfigSHA != "cfgsha" || p.Stage != (Stage{64, 64, 9, 8}) || p.InnerSteps != 10 ||
			p.BatchSize != 4 || p.GradAccum != 2 || p.LR != 1e-4 || p.WarmupSteps != 2 || p.DeltaDtype != "float16" ||
			p.BaseVersionID != "v-init" || p.Seed != 0*1000003+hdSlot(0, 0, 2, i) {
			t.Fatalf("params %+v", p)
		}
		if !j.trusted || j.vram != 5500 {
			t.Fatalf("trusted=%v vram=%d", j.trusted, j.vram)
		}
	}
	if js[0].params.UpdateID == js[1].params.UpdateID {
		t.Fatal("update ids collide")
	}
}

func mustShards(t *testing.T, f *hdRunFixture) []datasetShard {
	sh, err := f.e.s.datasetShards(f.e.ctx, f.e.s.d.DB, f.dataset.ID)
	if err != nil {
		t.Fatal(err)
	}
	return sh
}

func TestCreateHDRunRejectsMismatches(t *testing.T) {
	f := newRunFixture(t, 2, "off")
	e := f.e
	mk := func(mut func(*RunConfig)) error {
		cfg := RunConfig{HDFields: HDFields{Family: "hd", HDConfig: "nivra_hd_tiny", DatasetID: f.dataset.ID, InitVersionID: *f.run.versionPtr(), EvalGate: "off", ShardsPerJob: 2}}
		mut(&cfg)
		_, err := e.s.CreateRun(e.ctx, core.System, e.project, "x", cfg, "")
		return err
	}
	if err := mk(func(c *RunConfig) {}); err != nil {
		t.Fatalf("baseline config should be fine: %v", err)
	}
	for name, mut := range map[string]func(*RunConfig){
		"other vae":      func(c *RunConfig) { c.VAEVersionID = *f.run.versionPtr() },
		"other text enc": func(c *RunConfig) { c.TextEncoderID = "different" },
		"sha mismatch":   func(c *RunConfig) { c.HDConfigSHA = "nope" },
		"stage mismatch": func(c *RunConfig) { c.Stage = &Stage{128, 128, 9, 8} },
		"too many shards": func(c *RunConfig) {
			c.ShardsPerJob = 6
		},
		"gate needs val": func(c *RunConfig) { c.EvalGate = "pi"; c.DatasetID = "00000000-0000-4000-8000-000000000000" },
		"untrusted":      func(c *RunConfig) { c.AllowUntrusted = true },
		"vae target on latents": func(c *RunConfig) {
			c.Target = "vae"
		},
	} {
		if err := mk(mut); !errors.Is(err, ErrBadConfig) {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

func (r *Run) versionPtr() *string { return r.initVersion() }

// initVersion finds the run's init version id from its config.
func (r *Run) initVersion() *string { v := r.Config.InitVersionID; return &v }

func TestValidateHDTrain(t *testing.T) {
	f := newRunFixture(t, 3, "off")
	f.step()
	e := f.e
	dev := e.device(e.project, "gpu1")
	js := f.jobs()
	run, _ := e.s.runByID(e.ctx, f.run.ID)

	attempt := func(job jobRow) string {
		var att string
		if err := e.s.d.DB.QueryRow(e.ctx, `INSERT INTO job_attempts (job_id, device_id, attempt_number, fencing_token, state, lease_expires_at)
			VALUES ($1,$2,(SELECT count(*)+1 FROM job_attempts WHERE job_id=$1),nextval('fencing_seq'),'validating', now() + interval '1 hour') RETURNING id::text`,
			job.id, dev).Scan(&att); err != nil {
			t.Fatal(err)
		}
		return att
	}
	submit := func(job jobRow, att string, weights []byte, mut func(*hdUpdateManifest)) (scheduler.Verdict, error) {
		p := job.params
		m := hdUpdateManifest{Schema: "nivra.update/2", RunID: p.RunID, Round: p.Round, RoundBatch: p.RoundBatch, BaseVersionID: p.BaseVersionID,
			UpdateID: p.UpdateID, AttemptID: att, Target: p.Target, HDConfigSHA: p.HDConfigSHA, Stage: p.Stage, InnerSteps: p.InnerSteps, LR: p.LR,
			Shards: append([]shardRef{}, p.Shards...), SamplesSeen: int64(p.InnerSteps * p.BatchSize * p.GradAccum), DeltaDtype: p.DeltaDtype}
		if mut != nil {
			mut(&m)
		}
		mb, _ := json.Marshal(m)
		pb, _ := json.Marshal(p)
		w := e.art(e.project, "update_weights", weights)
		u := e.art(e.project, "update", mb)
		return e.s.validateHDTrain(e.ctx, scheduler.ValidationInput{JobID: job.id, AttemptID: att, ProjectID: e.project, DeviceID: dev, JobType: "ai.hd_train",
			Params: pb, Result: []byte(`{"loss_first":1.0,"loss_last":0.9,"samples_seen":160}`), Artifacts: map[string]string{"update_weights": w.ID, "update": u.ID},
			ModelVersion: run.ModelVersion, TrainingRun: &run.ID})
	}
	delta := func(v float32) []byte {
		return stBytes([]stFix{{"a", "F16", []int64{4, 4}, fill16(16, v)}, {"b", "F16", []int64{8}, fill16(8, v)}}, 0)
	}
	// bound = c*H*lr = 3*10*1e-4 = 3e-3
	v, err := submit(js[0], attempt(js[0]), delta(0.001), nil)
	if err != nil || !v.Accept || v.Detail["l2_norm"] == nil {
		t.Fatalf("good update: %+v %v", v, err)
	}
	// Idempotent re-validation of the same attempt stays accepted.
	var accAtt string
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT attempt_id::text FROM training_updates WHERE update_id=$1`, js[0].params.UpdateID).Scan(&accAtt)
	if accAtt == "" {
		t.Fatal("claim row missing")
	}
	if v, err := submit(js[0], accAtt, delta(0.001), func(m *hdUpdateManifest) {}); err != nil || !v.Accept {
		t.Fatalf("re-validation of the same attempt: %+v %v", v, err)
	}
	// Duplicate content in the same round (another job, byte-identical delta).
	if v, _ := submit(js[1], attempt(js[1]), delta(0.001), nil); v.Accept || !strings.Contains(v.Reason, "duplicate") {
		t.Fatalf("duplicate sha accepted: %+v", v)
	}
	// Echo mismatches.
	if v, _ := submit(js[1], attempt(js[1]), delta(0.0011), func(m *hdUpdateManifest) { m.BaseVersionID = "other" }); v.Accept || !strings.Contains(v.Reason, "base_version_id") {
		t.Fatalf("base mismatch: %+v", v)
	}
	if v, _ := submit(js[1], attempt(js[1]), delta(0.0012), func(m *hdUpdateManifest) { m.Shards[0].SHA256 = "00" }); v.Accept {
		t.Fatalf("shards echo mismatch accepted: %+v", v)
	}
	att := attempt(js[1])
	if v, _ := submit(js[1], att, delta(0.0013), func(m *hdUpdateManifest) { m.AttemptID = "someone-else" }); v.Accept {
		t.Fatalf("attempt echo mismatch accepted")
	}
	// Exact size: one extra byte, bound exceeded, wrong tensors.
	if v, _ := submit(js[1], attempt(js[1]), append(delta(0.0014), 0), nil); v.Accept {
		t.Fatal("padded update accepted")
	}
	if v, _ := submit(js[1], attempt(js[1]), delta(0.5), nil); v.Accept || !strings.Contains(v.Reason, "exceeds") {
		t.Fatalf("out-of-bound update accepted: %+v", v)
	}
	if v, _ := submit(js[1], attempt(js[1]), stBytes([]stFix{{"a", "F16", []int64{4, 4}, fill16(16, 0.001)}}, 0), nil); v.Accept {
		t.Fatal("missing tensor accepted")
	}
	// A genuine different update for job 1 is still accepted.
	if v, err := submit(js[1], attempt(js[1]), delta(0.0015), nil); err != nil || !v.Accept {
		t.Fatalf("second good update: %+v %v", v, err)
	}
	// The DB unique index itself backs the check.
	_, err = e.s.d.DB.Exec(e.ctx, `INSERT INTO training_updates (run_id, update_id, job_id, attempt_id, device_id, base_version, round, state, content_sha256)
		SELECT run_id, 'dup-direct', job_id, $1, device_id, base_version, round, 'pending', content_sha256 FROM training_updates WHERE update_id=$2`, newAttempt(t, e, js[2], dev), js[0].params.UpdateID)
	if err == nil {
		t.Fatal("unique (run_id, round, content_sha256) index missing")
	}
	// Stale: a result for an older base version is rejected without retry.
	pb, _ := json.Marshal(js[2].params)
	old := newUUID()
	if v, _ := e.s.validateHDTrain(e.ctx, scheduler.ValidationInput{JobID: js[2].id, AttemptID: "x", ProjectID: e.project, Params: pb,
		Artifacts: map[string]string{"update": "a", "update_weights": "b"}, ModelVersion: &old, TrainingRun: &run.ID}); v.Accept || v.Retry {
		t.Fatalf("stale: %+v", v)
	}
	_ = protocol.ErrBadRequest
}

func newAttempt(t *testing.T, e *hdEnv, job jobRow, dev string) string {
	var att string
	if err := e.s.d.DB.QueryRow(e.ctx, `INSERT INTO job_attempts (job_id, device_id, attempt_number, fencing_token, state, lease_expires_at)
		VALUES ($1,$2,(SELECT count(*)+1 FROM job_attempts WHERE job_id=$1),nextval('fencing_seq'),'validating', now() + interval '1 hour') RETURNING id::text`,
		job.id, dev).Scan(&att); err != nil {
		t.Fatal(err)
	}
	return att
}

func TestRetentionKeepsLastAndEveryTenth(t *testing.T) {
	f := newRunFixture(t, 2, "off")
	e := f.e
	for r := 1; r <= 25; r++ {
		if _, err := e.s.d.DB.Exec(e.ctx, `INSERT INTO checkpoints (run_id, project_id, model_version, step, round, trigger, created_at)
			VALUES ($1,$2,$3,$4::bigint,$5::int,'round', now() + ($5::int * interval '1 second'))`, f.run.ID, e.project, *f.run.ModelVersion, int64(r), r); err != nil {
			t.Fatal(err)
		}
	}
	e.s.applyHDRetention(e.ctx, f.run.ID, &Retention{KeepLast: 5, KeepEvery: 10})
	rows, _ := e.s.d.DB.Query(e.ctx, `SELECT round FROM checkpoints WHERE run_id=$1 AND trigger='round' AND state='valid' ORDER BY round`, f.run.ID)
	var got []int
	for rows.Next() {
		var r int
		_ = rows.Scan(&r)
		got = append(got, r)
	}
	rows.Close()
	if want := []int{10, 20, 21, 22, 23, 24, 25}; !reflect.DeepEqual(got, want) {
		t.Fatalf("kept rounds %v want %v", got, want)
	}
}

func TestPromoteHDTwiceIsSafe(t *testing.T) {
	f := newRunFixture(t, 2, "off")
	e := f.e
	w := e.art(e.project, "model_weights", randBytes(32))
	tr := e.art(e.project, "model_train_weights", randBytes(32))
	m := e.art(e.project, "model_version", []byte(`{"version_id":"x"}`))
	pending := newUUID()
	o := &hdOutcome{Pending: pending, Weights: w.ID, Train: tr.ID, Manifest: m.ID, Man: []byte(`{"version_id":"x"}`), WeightsSHA: w.SHA256,
		Applied: map[string]bool{}, RejectedWhy: map[string]string{}}
	base := *f.run.ModelVersion
	for i := 0; i < 2; i++ {
		if err := e.s.promoteHD(e.ctx, f.run, base, o, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	var n int
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT count(*) FROM model_versions WHERE run_id=$1`, f.run.ID).Scan(&n)
	var round int
	var train *string
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT current_round FROM training_runs WHERE id=$1`, f.run.ID).Scan(&round)
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT train_weights_artifact::text FROM model_versions WHERE id=$1`, pending).Scan(&train)
	if n != 2 || round != 1 || train == nil || *train != tr.ID {
		t.Fatalf("versions=%d round=%d train=%v", n, round, train)
	}
}

func TestWorkerGateCandidateFlow(t *testing.T) {
	f := newRunFixture(t, 2, "worker")
	e := f.e
	dev := e.device(e.project, "gpu")
	newCand := func(run *Run) *hdCandidate {
		w := e.art(e.project, "model_weights", randBytes(32))
		tr := e.art(e.project, "model_train_weights", randBytes(32))
		m := e.art(e.project, "model_version", []byte(`{"version_id":"x"}`))
		o := &hdOutcome{Pending: newUUID(), Weights: w.ID, Train: tr.ID, Manifest: m.ID, Man: []byte(`{"version_id":"x"}`), WeightsSHA: w.SHA256,
			Sum: hdAggSummary{AggregatedCount: 1}, Applied: map[string]bool{}, RejectedWhy: map[string]string{}}
		base, err := e.s.hdVersionRow(e.ctx, e.project, *run.ModelVersion)
		if err != nil {
			t.Fatal(err)
		}
		if err := e.s.openCandidate(e.ctx, run, base, o); err != nil {
			t.Fatal(err)
		}
		c, err := e.s.evaluatingCandidate(e.ctx, run)
		if err != nil || c == nil {
			t.Fatalf("candidate: %v %v", c, err)
		}
		return c
	}
	finish := func(run *Run, version string, loss float64) {
		var id string
		if err := e.s.d.DB.QueryRow(e.ctx, `SELECT id::text FROM jobs WHERE idempotency_key=$1`, "hdeval:"+run.ID+":"+version).Scan(&id); err != nil {
			t.Fatalf("eval job for %s: %v", version, err)
		}
		e.completeJob(id, dev, "", nil, map[string]any{"loss": loss, "n": 8, "version_id": version})
	}
	run, _ := e.s.runByID(e.ctx, f.run.ID)
	c := newCand(run)
	if done, err := e.s.advanceCandidate(e.ctx, run, c); err != nil || done {
		t.Fatalf("first advance: done=%v err=%v", done, err)
	}
	var n int
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT count(*) FROM jobs WHERE job_type='ai.hd_eval' AND req_trusted AND req_vram_mb=5500`).Scan(&n)
	if n != 2 {
		t.Fatalf("want a base and a candidate eval job, got %d", n)
	}
	if _, err := e.s.advanceCandidate(e.ctx, run, c); err != nil { // idempotent: no new jobs
		t.Fatal(err)
	}
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT count(*) FROM jobs WHERE job_type='ai.hd_eval'`).Scan(&n)
	if n != 2 {
		t.Fatalf("eval jobs duplicated: %d", n)
	}
	finish(run, *run.ModelVersion, 1.0)
	finish(run, c.VersionID, 0.9)
	if done, err := e.s.advanceCandidate(e.ctx, run, c); err != nil || !done {
		t.Fatalf("accept: done=%v err=%v", done, err)
	}
	run, _ = e.s.runByID(e.ctx, f.run.ID)
	if run.Round != 1 || *run.ModelVersion != c.VersionID {
		t.Fatalf("not promoted: round=%d", run.Round)
	}
	// Round 2: the loss rises more than max_heldout_increase (10%): rejected, round retried.
	c2 := newCand(run)
	e.s.advanceCandidate(e.ctx, run, c2)
	finish(run, c2.VersionID, 1.5)
	if done, err := e.s.advanceCandidate(e.ctx, run, c2); err != nil || !done {
		t.Fatalf("reject: done=%v err=%v", done, err)
	}
	var st string
	var batch int
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT state FROM hd_candidates WHERE version_id=$1`, c2.VersionID).Scan(&st)
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT round_batch FROM training_runs WHERE id=$1`, run.ID).Scan(&batch)
	run, _ = e.s.runByID(e.ctx, f.run.ID)
	if st != "rejected" || batch != 1 || run.Round != 1 || *run.ModelVersion != c.VersionID {
		t.Fatalf("state=%s batch=%d round=%d", st, batch, run.Round)
	}
}
