package training

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"nivra/internal/controller/scheduler"
)

// ---- int8 delta format ("int8-rowscale/1") ----

type goldenMeta struct {
	Format        string             `json:"format"`
	Shapes        map[string][]int64 `json:"shapes"`
	FileBytes     int64              `json:"file_bytes"`
	ExpectedBytes int64              `json:"expected_bytes"`
	InnerSteps    int                `json:"inner_steps"`
	LR            float64            `json:"lr"`
	BoundC        float64            `json:"bound_c"`
	Slack         float64            `json:"slack"`
	L2            float64            `json:"l2"`
	Linf          float64            `json:"linf"`
	Tensors       map[string]struct {
		L2     float64   `json:"l2"`
		Linf   float64   `json:"linf"`
		Scales []float64 `json:"scales"`
		Q      []int     `json:"q"`
	} `json:"tensors"`
	SHA256 string `json:"sha256"`
}

func i8data(vals ...int8) []byte {
	out := make([]byte, len(vals))
	for i, v := range vals {
		out[i] = byte(v)
	}
	return out
}

func f32data(vals ...float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
	}
	return out
}

func fillI8(n int, v int8) []byte {
	vals := make([]int8, n)
	for i := range vals {
		vals[i] = v
	}
	return i8data(vals...)
}

// int8 update for the test base {a: [4,4], b: [8]}: a has 4 row scales, b one.
func int8Fix(qa, qb int8, sa, sb float32) []stFix {
	return []stFix{
		{"a", "I8", []int64{4, 4}, fillI8(16, qa)},
		{"a#scale", "F32", []int64{4}, f32data(sa, sa, sa, sa)},
		{"b", "I8", []int64{8}, fillI8(8, qb)},
		{"b#scale", "F32", []int64{1}, f32data(sb)},
	}
}

func headerOf(t *testing.T, fix []stFix, extra int) (string, *stHeader) {
	t.Helper()
	p := writeTemp(t, stBytes(fix, extra))
	h, err := readSafetensorsHeader(p)
	if err != nil {
		return p, nil
	}
	return p, h
}

func TestInt8GoldenFixtureSharedWithPython(t *testing.T) {
	dir := filepath.Join("..", "..", "..", "ai", "tests", "fixtures")
	raw, err := os.ReadFile(filepath.Join(dir, "hd_int8_golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	var g goldenMeta
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	if g.Format != "int8-rowscale/1" || g.Slack != int8Slack {
		t.Fatalf("format/slack drifted: %+v (Go slack %v)", g.Format, int8Slack)
	}
	fx := filepath.Join(dir, "hd_int8_golden.safetensors")
	h, err := readSafetensorsHeader(fx)
	if err != nil {
		t.Fatal(err)
	}
	if h.Size != g.FileBytes || h.Size != g.ExpectedBytes {
		t.Fatalf("size %d, python says file=%d expected=%d", h.Size, g.FileBytes, g.ExpectedBytes)
	}
	// base with the golden shapes (fp16 train copy)
	var base []stFix
	for name, shape := range g.Shapes {
		n := int64(1)
		for _, d := range shape {
			n *= d
		}
		base = append(base, stFix{name, "F16", shape, fill16(int(n), 0)})
	}
	bh, err := readSafetensorsHeader(writeTemp(t, stBytes(base, 0)))
	if err != nil {
		t.Fatal(err)
	}
	if why := checkDeltaStructureInt8(h, bh); why != "" {
		t.Fatal(why)
	}
	// the fp16 structure check must refuse it
	if checkDeltaStructure(h, bh) == "" {
		t.Fatal("int8 file accepted as fp16")
	}
	linfMax := g.BoundC * float64(g.InnerSteps) * g.LR
	st, why, err := scanDeltaInt8(fx, h, bh, linfMax)
	if err != nil || why != "" {
		t.Fatalf("scan: %q %v", why, err)
	}
	rel := func(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(math.Abs(b), 1e-300) }
	if !rel(st.L2, g.L2) || !rel(st.Linf, g.Linf) || !st.Nonzero {
		t.Fatalf("stats go (%v, %v) python (%v, %v)", st.L2, st.Linf, g.L2, g.Linf)
	}
	// dequantised bound check: just below the largest element must fail, with the slack it must pass
	if _, why, _ := scanDeltaInt8(fx, h, bh, g.Linf/int8Slack*0.99); !strings.Contains(why, "exceeds") {
		t.Fatalf("bound not enforced on dequantised values: %q", why)
	}
	if _, why, _ := scanDeltaInt8(fx, h, bh, g.Linf/int8Slack*1.0001); why != "" {
		t.Fatalf("slack not applied: %q", why)
	}
	// per-tensor dequantised values: q * scale in float32, row by row, equals the stored statistics
	f, _ := os.Open(fx)
	defer f.Close()
	for name, want := range g.Tensors {
		qt, sct := h.Tensors[name], h.Tensors[name+"#scale"]
		if int64(len(want.Scales)) != sct.numel() || int64(len(want.Q)) != qt.numel() {
			t.Fatalf("%s: counts", name)
		}
		raw := make([]byte, qt.numel())
		if _, err := f.ReadAt(raw, h.DataStart+qt.Begin); err != nil {
			t.Fatal(err)
		}
		sraw := make([]byte, 4*sct.numel())
		if _, err := f.ReadAt(sraw, h.DataStart+sct.Begin); err != nil {
			t.Fatal(err)
		}
		cols := len(raw) / len(want.Scales)
		var ss, mx float64
		for i, b := range raw {
			if int(int8(b)) != want.Q[i] {
				t.Fatalf("%s q[%d]: %d vs %d", name, i, int8(b), want.Q[i])
			}
			s := math.Float32frombits(binary.LittleEndian.Uint32(sraw[4*(i/cols):]))
			if float64(s) != want.Scales[i/cols] {
				t.Fatalf("%s scale %d", name, i/cols)
			}
			v := float64(float32(int8(b)) * s)
			ss += v * v
			mx = math.Max(mx, math.Abs(v))
		}
		if !rel(math.Sqrt(ss), want.L2) || !rel(mx, want.Linf) {
			t.Fatalf("%s: go (%v, %v) python (%v, %v)", name, math.Sqrt(ss), mx, want.L2, want.Linf)
		}
	}
}

func TestInt8StructureAndScan(t *testing.T) {
	bh, err := readSafetensorsHeader(writeTemp(t, stBytes(fixBase(), 0)))
	if err != nil {
		t.Fatal(err)
	}
	good := int8Fix(3, -3, 0.0003, 0.0003) // |value| = 9e-4
	gp, gh := headerOf(t, good, 0)
	if gh == nil {
		t.Fatal("good fixture unreadable")
	}
	if why := checkDeltaStructureInt8(gh, bh); why != "" {
		t.Fatal(why)
	}
	st, why, err := scanDeltaInt8(gp, gh, bh, 0.003)
	if err != nil || why != "" || !st.Nonzero || math.Abs(st.Linf-9e-4) > 1e-9 {
		t.Fatalf("good scan: %+v %q %v", st, why, err)
	}
	wantL2 := math.Sqrt(24 * 9e-4 * 9e-4)
	if math.Abs(st.L2-wantL2) > 1e-7 {
		t.Fatalf("l2 %v want %v", st.L2, wantL2)
	}
	// exact size: one trailing byte breaks the tiling
	if _, h := headerOf(t, good, 1); h != nil {
		t.Fatal("padded file accepted")
	}
	// structural rejections
	structure := map[string][]stFix{
		"fp16 payload":     {{"a", "F16", []int64{4, 4}, fill16(16, 0.001)}, {"b", "F16", []int64{8}, fill16(8, 0.001)}},
		"missing scales":   {good[0], good[2], good[3]},
		"extra tensor":     append(append([]stFix{}, good...), stFix{"c", "I8", []int64{1}, i8data(1)}),
		"scale dtype":      {good[0], {"a#scale", "F16", []int64{4}, f16data(1, 1, 1, 1)}, good[2], good[3]},
		"scale count":      {good[0], {"a#scale", "F32", []int64{2}, f32data(1, 1)}, good[2], good[3]},
		"scale shape rank": {good[0], {"a#scale", "F32", []int64{2, 2}, f32data(1, 1, 1, 1)}, good[2], good[3]},
		"q shape":          {{"a", "I8", []int64{2, 8}, fillI8(16, 1)}, good[1], good[2], good[3]},
		"q dtype":          {{"a", "U8", []int64{4, 4}, fillI8(16, 1)}, good[1], good[2], good[3]},
		"renamed":          {good[0], good[1], {"z", "I8", []int64{8}, fillI8(8, 1)}, {"z#scale", "F32", []int64{1}, f32data(1)}},
	}
	for name, fix := range structure {
		_, h := headerOf(t, fix, 0)
		if h == nil {
			continue // the layout check already refused it
		}
		if checkDeltaStructureInt8(h, bh) == "" {
			t.Fatalf("%s accepted", name)
		}
	}
	// a base tensor named like a scale tensor can never be validated
	rb, _ := readSafetensorsHeader(writeTemp(t, stBytes([]stFix{{"x#scale", "F16", []int64{2}, fill16(2, 0)}}, 0)))
	if checkDeltaStructureInt8(gh, rb) == "" {
		t.Fatal("reserved suffix accepted in base")
	}
	// content rejections
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	content := map[string]struct {
		fix  []stFix
		want string
	}{
		"minus 128":        {int8Fix(-128, 3, 0.0003, 0.0003), "-128"},
		"negative scale":   {int8Fix(3, 3, -0.0003, 0.0003), "negative"},
		"nan scale":        {int8Fix(3, 3, nan, 0.0003), "NaN"},
		"inf scale":        {int8Fix(3, 3, 0.0003, inf), "Inf"},
		"zero scale, q!=0": {int8Fix(3, 3, 0, 0.0003), "zero scale"},
		"out of bound":     {int8Fix(100, 3, 0.0003, 0.0003), "exceeds"},
		"overflow":         {int8Fix(3, 3, 3e38, 0.0003), "overflows"},
		"all zeros":        {int8Fix(0, 0, 0, 0), "zeros"},
	}
	for name, c := range content {
		p, h := headerOf(t, c.fix, 0)
		if h == nil || checkDeltaStructureInt8(h, bh) != "" {
			t.Fatalf("%s: fixture should be structurally valid", name)
		}
		if _, why, _ := scanDeltaInt8(p, h, bh, 0.003); !strings.Contains(why, c.want) {
			t.Fatalf("%s: reason %q, want %q", name, why, c.want)
		}
	}
	// L2 bound per tensor: every element at the element bound is fine for L2 (bound * sqrt(n)), so only Linf triggers
	// first; a scale with q = 127 at the limit passes only with the slack
	p, h := headerOf(t, int8Fix(127, 127, 0.003/127, 0.003/127), 0)
	if _, why, _ := scanDeltaInt8(p, h, bh, 0.003); why != "" {
		t.Fatalf("value exactly at the bound rejected: %q", why)
	}
	// rows are independent: a zero row next to a non-zero row is fine
	zr := []stFix{{"a", "I8", []int64{4, 4}, append(fillI8(4, 0), fillI8(12, 5)...)}, {"a#scale", "F32", []int64{4}, f32data(0, 0.0002, 0.0002, 0.0002)},
		good[2], good[3]}
	p, h = headerOf(t, zr, 0)
	if _, why, _ := scanDeltaInt8(p, h, bh, 0.003); why != "" {
		t.Fatalf("zero row rejected: %q", why)
	}
}

func TestInt8RunConfigValidation(t *testing.T) {
	id := "11111111-1111-4111-8111-111111111111"
	mk := func(mut func(*RunConfig)) error {
		c := RunConfig{HDFields: HDFields{Family: "hd", HDConfig: "nivra_base", DatasetID: id, InitVersionID: id}}
		mut(&c)
		c.Defaults()
		return c.Validate()
	}
	if err := mk(func(c *RunConfig) { c.DeltaDtype = "int8" }); err != nil {
		t.Fatal(err)
	}
	if err := mk(func(c *RunConfig) { c.DeltaDtype = "int8"; c.OuterLRWarmupRounds = 5 }); err != nil {
		t.Fatal(err)
	}
	for name, mut := range map[string]func(*RunConfig){
		"float32":          func(c *RunConfig) { c.DeltaDtype = "float32" },
		"int4":             func(c *RunConfig) { c.DeltaDtype = "int4" },
		"negative warmup":  func(c *RunConfig) { c.OuterLRWarmupRounds = -1 },
		"start > 1":        func(c *RunConfig) { c.OuterLRWarmupRounds = 3; c.OuterLRWarmupStart = 1.5 },
		"start w/o rounds": func(c *RunConfig) { c.OuterLRWarmupStart = 0.5 },
	} {
		if mk(mut) == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	c := RunConfig{HDFields: HDFields{Family: "hd", HDConfig: "nivra_base", DatasetID: id, InitVersionID: id, OuterLRWarmupRounds: 4}}
	c.Defaults()
	if c.OuterLRWarmupStart != 0.3 || c.DeltaDtype != "float16" || warmupStart(c) != 0.3 {
		t.Fatalf("defaults: %+v", c)
	}
}

func TestValidateHDTrainInt8(t *testing.T) {
	f := newRunFixture(t, 2, "off", func(c *RunConfig) { c.DeltaDtype = "int8" })
	f.step()
	e := f.e
	dev := e.device(e.project, "gpu1")
	js := f.jobs()
	if len(js) != 2 || js[0].params.DeltaDtype != "int8" {
		t.Fatalf("jobs: %+v", js)
	}
	run, _ := e.s.runByID(e.ctx, f.run.ID)
	submit := func(job jobRow, weights []byte, mut func(*hdUpdateManifest)) scheduler.Verdict {
		att := newAttempt(t, e, job, dev)
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
		v, err := e.s.validateHDTrain(e.ctx, scheduler.ValidationInput{JobID: job.id, AttemptID: att, ProjectID: e.project, DeviceID: dev, JobType: "ai.hd_train",
			Params: pb, Result: []byte(`{"loss_first":1.0,"loss_last":0.9}`), Artifacts: map[string]string{"update_weights": w.ID, "update": u.ID},
			ModelVersion: run.ModelVersion, TrainingRun: &run.ID})
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	// bound = c*H*lr = 3*10*1e-4 = 3e-3
	good := stBytes(int8Fix(3, -3, 0.0003, 0.0003), 0)
	if v := submit(js[0], good, nil); !v.Accept || v.Detail["l2_norm"] == nil {
		t.Fatalf("good int8 update: %+v", v)
	}
	want := math.Sqrt(24 * 9e-4 * 9e-4)
	var l2 float64
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT (metadata->>'l2_norm')::float8 FROM training_updates WHERE update_id=$1`, js[0].params.UpdateID).Scan(&l2)
	if math.Abs(l2-want) > 1e-6 {
		t.Fatalf("recorded l2 %v want %v", l2, want)
	}
	// a float16 payload for an int8 run, and the other way round, are structure errors
	fp16 := stBytes([]stFix{{"a", "F16", []int64{4, 4}, fill16(16, 0.001)}, {"b", "F16", []int64{8}, fill16(8, 0.001)}}, 0)
	if v := submit(js[1], fp16, nil); v.Accept || !strings.Contains(v.Reason, "update_weights rejected") {
		t.Fatalf("fp16 payload accepted for int8 run: %+v", v)
	}
	if v := submit(js[1], append(stBytes(int8Fix(2, -2, 0.0003, 0.0003), 0), 0), nil); v.Accept {
		t.Fatal("padded int8 update accepted")
	}
	if v := submit(js[1], stBytes(int8Fix(-128, 2, 0.0003, 0.0003), 0), nil); v.Accept || !strings.Contains(v.Reason, "-128") {
		t.Fatalf("-128 accepted: %+v", v)
	}
	if v := submit(js[1], stBytes(int8Fix(100, 2, 0.0003, 0.0003), 0), nil); v.Accept || !strings.Contains(v.Reason, "exceeds") {
		t.Fatalf("out-of-bound dequantised values accepted: %+v", v)
	}
	// the manifest must say int8
	if v := submit(js[1], stBytes(int8Fix(2, -2, 0.0003, 0.0003), 0), func(m *hdUpdateManifest) { m.DeltaDtype = "float16" }); v.Accept || !strings.Contains(v.Reason, "delta_dtype") {
		t.Fatalf("delta_dtype echo: %+v", v)
	}
	if v := submit(js[1], stBytes(int8Fix(2, -2, 0.0003, 0.0003), 0), nil); !v.Accept {
		t.Fatalf("second good update: %+v", v)
	}
}
