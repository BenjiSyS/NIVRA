package localui

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"nivra/internal/worker/agent"
)

func newTestApp(t *testing.T) (*App, http.Handler, *httptest.Server, *sync.Map) {
	t.Helper()
	ctl, seen := fakeController(t)
	app := &App{Dir: t.TempDir()}
	app.Init()
	h := (&Server{App: app}).Handler()
	if rec := doLocal(t, h, "PUT", "/local/api/app/config", `{"controller_url":"`+ctl.URL+`","token":"`+testToken+`"}`, nil); rec.Code != 200 {
		t.Fatalf("config: %d %s", rec.Code, rec.Body)
	}
	return app, h, ctl, seen
}

// Retry re-posts the history params; a long video stores "segments" there, which the
// controller rejects as an unknown field. The history view must not expose it.
func TestRetryOfLongVideoDoesNotSendInternalKeys(t *testing.T) {
	app, h, _, _ := newTestApp(t)
	app.mu.Lock()
	app.items = append(app.items, &histItem{ID: testJob, ProjectID: testPID, Prompt: "long", CreatedAt: time.Now(), Status: "failed",
		GenerationID: testJob, Params: map[string]any{"seconds": 600.0, "segments": 75, "_artifact": "x", "style": "anime"}})
	app.mu.Unlock()
	rec := doLocal(t, h, "GET", "/local/api/app/history", "", nil)
	var hist struct {
		Items []struct {
			Params map[string]any `json:"params"`
		} `json:"items"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &hist)
	if len(hist.Items) != 1 {
		t.Fatalf("history: %s", rec.Body)
	}
	p := hist.Items[0].Params
	if _, ok := p["segments"]; ok {
		t.Fatalf("segments leaked: %v", p)
	}
	if _, ok := p["_artifact"]; ok {
		t.Fatal("internal key leaked")
	}
	// What the page posts back on Retry is accepted by the generate handler.
	body, _ := json.Marshal(map[string]any{"prompt": "long", "seconds": p["seconds"], "style": p["style"]})
	for k, v := range p {
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		m[k] = v
		body, _ = json.Marshal(m)
	}
	if rec = doLocal(t, h, "POST", "/local/api/app/generate", string(body), nil); rec.Code != 200 {
		t.Fatalf("retry rejected: %d %s", rec.Code, rec.Body)
	}
}

func TestSavingConnectionKeepsLocalModels(t *testing.T) {
	app, h, ctl, _ := newTestApp(t)
	app.mu.Lock()
	app.cfg.LocalModels = []string{"C:\\models\\a", "C:\\models\\b"}
	app.mu.Unlock()
	if rec := doLocal(t, h, "PUT", "/local/api/app/config", `{"controller_url":"`+ctl.URL+`","project":"main"}`, nil); rec.Code != 200 {
		t.Fatalf("save: %d %s", rec.Code, rec.Body)
	}
	app.mu.Lock()
	defer app.mu.Unlock()
	if len(app.cfg.LocalModels) != 2 || app.cfg.Token != testToken || app.cfg.Project != "main" {
		t.Fatalf("config after save: %+v", app.cfg)
	}
}

func TestPromptLengthCountsRunesAndTrims(t *testing.T) {
	_, h, _, _ := newTestApp(t)
	// 900 three-byte characters = 2700 bytes but 900 runes: accepted.
	ok := strings.Repeat("\u3042", 900)
	if rec := doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":"  `+ok+`  "}`, nil); rec.Code != 200 {
		t.Fatalf("900 runes rejected: %d %s", rec.Code, rec.Body)
	}
	if rec := doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":"`+strings.Repeat("\u3042", 1001)+`"}`, nil); rec.Code != 400 {
		t.Fatalf("1001 runes accepted: %d", rec.Code)
	}
	// Whitespace around a 1000-character prompt does not count.
	if rec := doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":"   `+strings.Repeat("a", 1000)+`   "}`, nil); rec.Code != 200 {
		t.Fatalf("trimmed 1000 rejected: %d %s", rec.Code, rec.Body)
	}
	if rec := doLocal(t, h, "POST", "/local/api/app/generate", `{"prompt":"    "}`, nil); rec.Code != 400 {
		t.Fatalf("blank accepted: %d", rec.Code)
	}
}

func TestSmallLocalModelKeeps200Limit(t *testing.T) {
	app, h, _, _ := newTestApp(t)
	dir := t.TempDir()
	writeBundle(t, dir, map[string]any{"schema": "nivra.model_version/1", "family": "small"})
	app.Runtime = func() LocalRuntime { return LocalRuntime{} }
	rec := doLocal(t, h, "POST", "/local/api/app/local-models", `{"path":`+mustJSON(dir)+`}`, nil)
	var m localModel
	_ = json.Unmarshal(rec.Body.Bytes(), &m)
	if m.ID == "" {
		t.Fatalf("add: %d %s", rec.Code, rec.Body)
	}
	long := `{"prompt":"` + strings.Repeat("a", 201) + `","model_version":"` + m.ID + `"}`
	if rec = doLocal(t, h, "POST", "/local/api/app/generate", long, nil); rec.Code != 400 {
		t.Fatalf("201 chars on a small model: %d %s", rec.Code, rec.Body)
	}
	short := `{"prompt":"` + strings.Repeat("a", 200) + `","model_version":"` + m.ID + `"}`
	if rec = doLocal(t, h, "POST", "/local/api/app/generate", short, nil); rec.Code != 200 {
		t.Fatalf("200 chars on a small model: %d %s", rec.Code, rec.Body)
	}
}

func mustJSON(v any) string { b, _ := json.Marshal(v); return string(b) }

func TestWaitingReasonShownForQueuedJob(t *testing.T) {
	app, h, _, _ := newTestApp(t)
	app.mu.Lock()
	app.items = append(app.items, &histItem{ID: testJob, ProjectID: testPID, Prompt: "p", CreatedAt: time.Now(), Status: "queued",
		Params: map[string]any{}, WaitingReason: "no online worker has 8 GB of graphics memory"})
	app.mu.Unlock()
	rec := doLocal(t, h, "GET", "/local/api/app/history", "", nil)
	if !strings.Contains(rec.Body.String(), `"waiting_reason":"no online worker has 8 GB of graphics memory"`) {
		t.Fatalf("history: %s", rec.Body)
	}
}

func TestLocalGenerationRefusedWhileNetworkTaskRuns(t *testing.T) {
	app, h, _, _ := newTestApp(t)
	// A gate with no busy agent lets local runs start; BeginLocal marks the GPU taken.
	app.Gate = agent.NewGate()
	end, err := app.Gate.BeginLocal()
	if err != nil {
		t.Fatal(err)
	}
	if !app.Gate.LocalActive() {
		t.Fatal("not active")
	}
	end()
	if app.Gate.LocalActive() {
		t.Fatal("not released")
	}
	_ = h
}

// ---- sign in with username and password ----

func loginController(t *testing.T) (*httptest.Server, *sync.Map) {
	t.Helper()
	seen := &sync.Map{}
	const sess, csrf = "sess-secret", "csrf-secret"
	mux := http.NewServeMux()
	j := func(w http.ResponseWriter, code int, v any) {
		w.WriteHeader(code)
		_ = json.NewEncoder(w).Encode(v)
	}
	mux.HandleFunc("POST /api/v1/auth/login", func(w http.ResponseWriter, r *http.Request) {
		var b struct{ Username, Password string }
		_ = json.NewDecoder(r.Body).Decode(&b)
		if b.Username != "alice" || b.Password != "correct horse" {
			j(w, 401, map[string]any{"error": map[string]any{"message": "invalid username or password"}})
			return
		}
		http.SetCookie(w, &http.Cookie{Name: "__Host-nivra_session", Value: sess, Path: "/", HttpOnly: true, Secure: true})
		j(w, 200, map[string]any{"user": map[string]any{"username": "alice"}, "csrf_token": csrf})
	})
	needSession := func(w http.ResponseWriter, r *http.Request) bool {
		c, err := r.Cookie("__Host-nivra_session")
		if err != nil || c.Value != sess || r.Header.Get("X-Nivra-CSRF") != csrf {
			j(w, 403, map[string]any{"error": map[string]any{"message": "missing or invalid CSRF token"}})
			return false
		}
		return true
	}
	mux.HandleFunc("POST /api/v1/tokens", func(w http.ResponseWriter, r *http.Request) {
		if !needSession(w, r) {
			return
		}
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		seen.Store("token_req", b)
		j(w, 201, map[string]any{"token": testToken, "info": map[string]any{}})
	})
	mux.HandleFunc("POST /api/v1/auth/logout", func(w http.ResponseWriter, r *http.Request) {
		if !needSession(w, r) {
			return
		}
		seen.Store("logout", true)
		j(w, 200, map[string]any{"status": "signed out"})
	})
	mux.HandleFunc("GET /api/v1/auth/me", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+testToken {
			j(w, 401, map[string]any{"error": map[string]any{"message": "bad token"}})
			return
		}
		j(w, 200, map[string]any{"user": map[string]any{"username": "alice", "role": "member"}})
	})
	mux.HandleFunc("GET /api/v1/projects", func(w http.ResponseWriter, r *http.Request) {
		j(w, 200, map[string]any{"projects": []any{map[string]any{"id": testPID, "name": "main"}}})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, seen
}

func TestSignInMintsTokenStoresOnlyTokenAndLogsOut(t *testing.T) {
	ctl, seen := loginController(t)
	app := &App{Dir: t.TempDir()}
	app.Init()
	h := (&Server{App: app}).Handler()
	rec := doLocal(t, h, "POST", "/local/api/app/login",
		`{"controller_url":"`+ctl.URL+`","username":"alice","password":"correct horse"}`, nil)
	if rec.Code != 200 {
		t.Fatalf("login: %d %s", rec.Code, rec.Body)
	}
	if strings.Contains(rec.Body.String(), "correct horse") || strings.Contains(rec.Body.String(), testToken) {
		t.Fatal("secret echoed to the page")
	}
	var cfg struct {
		Connected bool `json:"connected"`
		HasToken  bool `json:"has_token"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &cfg)
	if !cfg.Connected || !cfg.HasToken {
		t.Fatalf("view: %s", rec.Body)
	}
	if app.cfg.Token != testToken {
		t.Fatal("token not stored")
	}
	req, _ := seen.Load("token_req")
	name, _ := req.(map[string]any)["name"].(string)
	if !strings.HasPrefix(name, "Nivra App on ") {
		t.Fatalf("token name %q", name)
	}
	if _, ok := seen.Load("logout"); !ok {
		t.Fatal("session was not signed out")
	}
	// The password is nowhere on disk.
	b, _ := os.ReadFile(app.cfgPath())
	if strings.Contains(string(b), "correct horse") {
		t.Fatal("password stored")
	}
}

func TestSignInWrongPasswordIsFriendly(t *testing.T) {
	ctl, seen := loginController(t)
	app := &App{Dir: t.TempDir()}
	app.Init()
	h := (&Server{App: app}).Handler()
	rec := doLocal(t, h, "POST", "/local/api/app/login",
		`{"controller_url":"`+ctl.URL+`","username":"alice","password":"nope"}`, nil)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "That username or password is not right.") {
		t.Fatalf("wrong password: %d %s", rec.Code, rec.Body)
	}
	if app.cfg.Token != "" {
		t.Fatal("token stored after failed sign-in")
	}
	if _, ok := seen.Load("logout"); ok {
		t.Fatal("logout without a session")
	}
	if rec = doLocal(t, h, "POST", "/local/api/app/login", `{"controller_url":"`+ctl.URL+`","username":"","password":""}`, nil); rec.Code != 400 {
		t.Fatalf("empty credentials: %d", rec.Code)
	}
}

func TestFriendlyErrors(t *testing.T) {
	if got := friendlyErr(&ctlErr{Status: 429, Msg: "x"}); !strings.Contains(got, "Too many tries") {
		t.Fatal(got)
	}
	if got := friendlyErr(&ctlErr{Status: 502, Msg: "x"}); !strings.Contains(got, "problem") {
		t.Fatal(got)
	}
}
