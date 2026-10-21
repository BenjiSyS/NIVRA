package localui

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

const (
	testToken = "nivra_pat_test"
	testJob   = "11111111-2222-3333-4444-555555555555"
	testPID   = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
)

// fakeController mimics the controller routes the app uses.
func fakeController(t *testing.T) (*httptest.Server, *sync.Map) {
	seen := &sync.Map{}
	polls := 0
	var mu sync.Mutex
	mux := http.NewServeMux()
	auth := func(h http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") != "Bearer "+testToken {
				w.WriteHeader(401)
				_, _ = w.Write([]byte(`{"error":{"code":"unauthorized","message":"bad token"}}`))
				return
			}
			h(w, r)
		}
	}
	j := func(w http.ResponseWriter, v any) { _ = json.NewEncoder(w).Encode(v) }
	mux.HandleFunc("GET /api/v1/auth/me", auth(func(w http.ResponseWriter, r *http.Request) {
		j(w, map[string]any{"user": map[string]any{"username": "alice", "role": "operator"}})
	}))
	mux.HandleFunc("GET /api/v1/projects", auth(func(w http.ResponseWriter, r *http.Request) {
		j(w, map[string]any{"projects": []any{map[string]any{"id": testPID, "name": "main"}}})
	}))
	mux.HandleFunc("GET /api/v1/projects/{pid}/models", auth(func(w http.ResponseWriter, r *http.Request) {
		j(w, map[string]any{"models": []any{
			map[string]any{"id": "m1", "run_name": "Nivra Medium", "architecture": "nivra-dit-medium", "family": "hd", "active": true, "parameter_count": 932333696},
			map[string]any{"id": "m2", "run_name": nil, "architecture": "nivra-small", "family": "", "active": false, "parameter_count": 2410000},
		}})
	}))
	mux.HandleFunc("POST /api/v1/projects/{pid}/generate", auth(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		seen.Store("generate", body)
		j(w, map[string]any{"job_id": testJob, "model_version": "m1"})
	}))
	mux.HandleFunc("GET /api/v1/projects/{pid}/jobs/{jid}", auth(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		polls++
		n := polls
		mu.Unlock()
		if n < 2 {
			j(w, map[string]any{"status": "running", "current_attempt": map[string]any{"device_name": "gpu-pc", "progress": 0.5, "stage": "sampling"}})
			return
		}
		j(w, map[string]any{"status": "completed", "result": map[string]any{"artifact_id": "art1"}})
	}))
	mux.HandleFunc("GET /api/v1/projects/{pid}/artifacts/{aid}", auth(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("0123456789fake-mp4"))
	}))
	mux.HandleFunc("POST /api/v1/projects/{pid}/jobs/{jid}/cancel", auth(func(w http.ResponseWriter, r *http.Request) {
		seen.Store("cancel", r.PathValue("jid"))
		j(w, map[string]any{"ok": true})
	}))
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, seen
}

func doLocal(t *testing.T, h http.Handler, method, path, body string, hdr map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, "http://127.0.0.1:8741"+path, strings.NewReader(body))
	req.Host = "127.0.0.1:8741"
	if method != "GET" {
		req.Header.Set("X-Nivra-Local", "1")
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestAppFlow(t *testing.T) {
	ctl, seen := fakeController(t)
	dir := t.TempDir()
	app := &App{Dir: dir}
	app.Init()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h := (&Server{App: app}).Handler()

	// Not configured yet: generation is refused.
	if rec := doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":"x"}`, nil); rec.Code != 409 {
		t.Fatalf("generate before config: %d %s", rec.Code, rec.Body)
	}
	// A wrong token is rejected and not stored.
	rec := doLocal(t, h, "PUT", "/local/api/app/config", `{"controller_url":"`+ctl.URL+`","token":"wrong"}`, nil)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "did not accept") {
		t.Fatalf("bad token: %d %s", rec.Code, rec.Body)
	}
	rec = doLocal(t, h, "PUT", "/local/api/app/config", `{"controller_url":"`+ctl.URL+`","token":"`+testToken+`"}`, nil)
	if rec.Code != 200 {
		t.Fatalf("config: %d %s", rec.Code, rec.Body)
	}
	if strings.Contains(rec.Body.String(), testToken) {
		t.Fatal("token echoed to the page")
	}
	var cfg struct {
		Connected bool `json:"connected"`
		HasToken  bool `json:"has_token"`
		Project   struct{ Name string }
		User      struct{ Username string }
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &cfg)
	if !cfg.Connected || !cfg.HasToken || cfg.Project.Name != "main" || cfg.User.Username != "alice" {
		t.Fatalf("config view: %s", rec.Body)
	}
	if st, err := os.Stat(filepath.Join(dir, "app.json")); err != nil || st.Size() == 0 {
		t.Fatalf("app.json not saved: %v", err)
	}

	rec = doLocal(t, h, "GET", "/local/api/app/models", "", nil)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"family":"hd"`) || !strings.Contains(rec.Body.String(), `"name":"nivra-small"`) {
		t.Fatalf("models: %d %s", rec.Code, rec.Body)
	}

	// Unknown fields are rejected; a valid request is forwarded with a seed.
	if rec = doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":"x","evil":1}`, nil); rec.Code != 400 {
		t.Fatalf("unknown field: %d", rec.Code)
	}
	rec = doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":" a cat in fog ","width":1280,"height":720,"seconds":8,"sampler":"ab2"}`, nil)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), testJob) {
		t.Fatalf("generate: %d %s", rec.Code, rec.Body)
	}
	b, _ := seen.Load("generate")
	body := b.(map[string]any)
	if body["prompt"] != "a cat in fog" || body["width"] != float64(1280) || body["seconds"] != float64(8) || body["seed"] == nil {
		t.Fatalf("forwarded body: %v", body)
	}

	// The poller moves the item running -> downloading -> done.
	go app.Run(ctx)
	deadline := time.Now().Add(15 * time.Second)
	var items []map[string]any
	for time.Now().Before(deadline) {
		rec = doLocal(t, h, "GET", "/local/api/app/history", "", nil)
		var hist struct{ Items []map[string]any }
		_ = json.Unmarshal(rec.Body.Bytes(), &hist)
		items = hist.Items
		if len(items) == 1 && items[0]["status"] == "done" {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if len(items) != 1 || items[0]["status"] != "done" || items[0]["video_url"] != "/local/api/app/videos/"+testJob+".mp4" {
		t.Fatalf("history: %v", items)
	}
	if _, ok := items[0]["params"].(map[string]any)["_artifact"]; ok {
		t.Fatal("internal params leaked")
	}

	rec = doLocal(t, h, "GET", "/local/api/app/videos/"+testJob+".mp4", "", map[string]string{"Range": "bytes=0-3"})
	if rec.Code != 206 || rec.Body.String() != "0123" || rec.Header().Get("Content-Type") != "video/mp4" {
		t.Fatalf("video range: %d %q %s", rec.Code, rec.Body, rec.Header().Get("Content-Type"))
	}
	if rec = doLocal(t, h, "GET", "/local/api/app/videos/..%2Fapp.json", "", nil); rec.Code != 404 {
		t.Fatalf("path traversal: %d", rec.Code)
	}

	// History survives a restart.
	app2 := &App{Dir: dir}
	app2.Init()
	if len(app2.items) != 1 || app2.items[0].Status != "done" || app2.cfg.Token != testToken {
		t.Fatalf("reload: %+v", app2.items)
	}

	rec = doLocal(t, h, "DELETE", "/local/api/app/history/"+testJob, "", nil)
	if rec.Code != 200 {
		t.Fatalf("delete: %d", rec.Code)
	}
	if _, err := os.Stat(app.videoPath(testJob)); !os.IsNotExist(err) {
		t.Fatal("video not removed")
	}
}

func TestAppCancelAndGuards(t *testing.T) {
	ctl, seen := fakeController(t)
	app := &App{Dir: t.TempDir()}
	app.Init()
	h := (&Server{App: app}).Handler()
	if rec := doLocal(t, h, "PUT", "/local/api/app/config", `{"controller_url":"`+ctl.URL+`","token":"`+testToken+`"}`, nil); rec.Code != 200 {
		t.Fatalf("config: %d %s", rec.Code, rec.Body)
	}
	if rec := doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":"x"}`, nil); rec.Code != 200 {
		t.Fatalf("generate: %d", rec.Code)
	}
	rec := doLocal(t, h, "POST", "/local/api/app/history/"+testJob+"/cancel", "", nil)
	if rec.Code != 200 {
		t.Fatalf("cancel: %d %s", rec.Code, rec.Body)
	}
	if v, _ := seen.Load("cancel"); v != testJob {
		t.Fatal("cancel not forwarded")
	}
	if app.items[0].Status != "cancelled" {
		t.Fatalf("status %s", app.items[0].Status)
	}
	// State-changing calls need the local header and a same-origin Origin.
	req := httptest.NewRequest("POST", "http://127.0.0.1:8741/local/api/app/generate", strings.NewReader(`{"prompt":"x"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 403 {
		t.Fatalf("missing header: %d", rec.Code)
	}
	rec = doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":"x"}`, map[string]string{"Origin": "https://evil.example"})
	if rec.Code != 403 {
		t.Fatalf("cross origin: %d", rec.Code)
	}
	req = httptest.NewRequest("GET", "http://evil.example/local/api/app/config", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != 403 {
		t.Fatalf("rebinding host: %d", rec.Code)
	}
	if rec = doLocal(t, h, "POST", "/local/api/app/quit", "", nil); rec.Code != 400 {
		t.Fatalf("quit without Quit: %d", rec.Code)
	}
	_, _ = io.Discard.Write(nil)
}
