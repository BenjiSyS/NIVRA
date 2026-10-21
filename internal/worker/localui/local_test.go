package localui

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestETA(t *testing.T) {
	t0 := time.Now().Add(-100 * time.Second)
	h := &histItem{Status: "running"}
	if h.eta(time.Now()) != -1 {
		t.Fatal("no samples must be unknown")
	}
	// 25% done after 100 s at a steady rate: 300 s left.
	h.samples = []progSample{{t0, 0}, {t0.Add(100 * time.Second), 0.25}}
	if e := h.eta(t0.Add(100 * time.Second)); e < 295 || e > 305 {
		t.Fatalf("eta %d, want ~300", e)
	}
	// Time since the last update counts down.
	if e := h.eta(t0.Add(160 * time.Second)); e < 235 || e > 245 {
		t.Fatalf("eta %d, want ~240", e)
	}
	h.Status = "queued"
	if h.eta(time.Now()) != -1 {
		t.Fatal("queued must be unknown")
	}
	h.Status = "done"
	if h.eta(time.Now()) != -1 {
		t.Fatal("finished must be unknown")
	}
}

func writeBundle(t *testing.T, dir string, man map[string]any) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(man)
	for name, data := range map[string][]byte{"model_version.json": b, "weights.safetensors": {1}, "vae.safetensors": {1}} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

func TestLocalModels(t *testing.T) {
	root := t.TempDir()
	good := filepath.Join(root, "models", "anime-v1")
	writeBundle(t, good, map[string]any{"schema": "nivra.model_version/1", "config_name": "nivra_medium", "family": "hd",
		"param_count": 932333696, "styles": []string{"anime", "3d animation"}, "default_style": "anime"})
	bad := filepath.Join(root, "models", "notes")
	_ = os.MkdirAll(bad, 0o700)
	m, err := inspectBundle(good)
	if err != nil || m.Family != "hd" || m.DefaultStyle != "anime" || !strings.HasPrefix(m.ID, localPrefix) || m.Name != "anime-v1 (nivra_medium)" {
		t.Fatalf("inspect: %+v %v", m, err)
	}
	if _, err := inspectBundle(bad); err == nil {
		t.Fatal("folder without manifest accepted")
	}
	if _, err := inspectBundle("relative/path"); err == nil {
		t.Fatal("relative path accepted")
	}
	// Missing VAE is reported.
	_ = os.Remove(filepath.Join(good, "vae.safetensors"))
	if _, err := inspectBundle(good); err == nil || !strings.Contains(err.Error(), "vae.safetensors") {
		t.Fatalf("missing vae: %v", err)
	}
	_ = os.WriteFile(filepath.Join(good, "vae.safetensors"), []byte{1}, 0o600)

	a := &App{Dir: t.TempDir(), ModelDirs: []string{filepath.Join(root, "models")}}
	a.Init()
	ms := a.localModels()
	if len(ms) != 1 || ms[0].Path != good {
		t.Fatalf("discovered %+v", ms)
	}
	h := (&Server{App: a}).Handler()
	other := filepath.Join(root, "elsewhere", "toon")
	writeBundle(t, other, map[string]any{"schema": "nivra.model_version/1", "family": "hd"})
	if rec := doLocal(t, h, "POST", "/local/api/app/local-models", `{"path":`+jsonString(other)+`}`, nil); rec.Code != 200 {
		t.Fatalf("add: %d %s", rec.Code, rec.Body)
	}
	rec := doLocal(t, h, "GET", "/local/api/app/models", "", nil)
	if rec.Code != 200 || strings.Count(rec.Body.String(), `"source":"local"`) != 2 || !strings.Contains(rec.Body.String(), "network_error") {
		t.Fatalf("models without controller: %d %s", rec.Code, rec.Body)
	}
	// Local generation without a runtime is refused cleanly.
	rec = doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":"x","model_version":"`+localID(other)+`"}`, nil)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "cannot run models locally") {
		t.Fatalf("no runtime: %d %s", rec.Code, rec.Body)
	}
	if rec = doLocal(t, h, "DELETE", "/local/api/app/local-models/"+strings.TrimPrefix(localID(other), localPrefix), "", nil); rec.Code != 200 {
		t.Fatalf("remove: %d", rec.Code)
	}
	if len(a.localModels()) != 1 {
		t.Fatal("remove did not forget the folder")
	}
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
