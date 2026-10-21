package localui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"nivra/internal/shared/config"
)

// App backs the Nivra App (chat-style generation) on the local UI server. It
// talks to the controller with the user's personal API token, which never
// leaves this process: the page only learns whether one is stored.
type App struct {
	Dir           string // data directory (app.json, app-history.json, videos/)
	AllowInsecure bool
	// DefaultController is used until the user picks a controller URL.
	DefaultController func() string
	Log               *slog.Logger
	// Quit, when set, is called by POST /local/api/app/quit (closes Nivra.exe).
	Quit func()

	mu        sync.Mutex
	cfg       appConfig
	items     []*histItem
	user      *appUser
	project   *appProject
	connected bool
	lastErr   string
	hc        *http.Client
	wake      chan struct{}
}

type appConfig struct {
	ControllerURL string `json:"controller_url"`
	Token         string `json:"token"`
	Project       string `json:"project"`
}

type appUser struct {
	Username string `json:"username"`
	Role     string `json:"role"`
}

type appProject struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type histItem struct {
	ID           string         `json:"id"`
	ProjectID    string         `json:"project_id"`
	Prompt       string         `json:"prompt"`
	Params       map[string]any `json:"params"`
	CreatedAt    time.Time      `json:"created_at"`
	FinishedAt   *time.Time     `json:"finished_at,omitempty"`
	Status       string         `json:"status"`
	Progress     float64        `json:"progress"`
	Stage        string         `json:"stage"`
	Worker       string         `json:"worker"`
	Error        string         `json:"error"`
	Video        bool           `json:"video"`
	pollFailures int
}

var jobIDRe = regexp.MustCompile(`^[0-9a-fA-F-]{36}$`)

func (a *App) cfgPath() string            { return filepath.Join(a.Dir, "app.json") }
func (a *App) histPath() string           { return filepath.Join(a.Dir, "app-history.json") }
func (a *App) videoDir() string           { return filepath.Join(a.Dir, "videos") }
func (a *App) videoPath(id string) string { return filepath.Join(a.videoDir(), id+".mp4") }

// Init loads stored settings and history.
func (a *App) Init() {
	a.hc = &http.Client{Timeout: 60 * time.Second}
	a.wake = make(chan struct{}, 1)
	if b, err := os.ReadFile(a.cfgPath()); err == nil {
		_ = json.Unmarshal(b, &a.cfg)
	}
	if b, err := os.ReadFile(a.histPath()); err == nil {
		_ = json.Unmarshal(b, &a.items)
	}
	if a.Log == nil {
		a.Log = slog.Default()
	}
}

// writeFileAtomic writes via a temp file so a crash never leaves a torn file.
func writeFileAtomic(path string, b []byte, perm os.FileMode) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, perm); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (a *App) saveCfgLocked() error {
	b, _ := json.MarshalIndent(a.cfg, "", "  ")
	return writeFileAtomic(a.cfgPath(), b, 0o600)
}

func (a *App) saveHistLocked() {
	b, _ := json.MarshalIndent(a.items, "", "  ")
	if err := writeFileAtomic(a.histPath(), b, 0o600); err != nil {
		a.Log.Warn("app history not saved", "err", err)
	}
}

func (a *App) controllerURL() string {
	if a.cfg.ControllerURL != "" {
		return a.cfg.ControllerURL
	}
	if a.DefaultController != nil {
		return a.DefaultController()
	}
	return ""
}

// ---- controller client ----

type ctlErr struct {
	Status int
	Msg    string
}

func (e *ctlErr) Error() string { return fmt.Sprintf("controller: HTTP %d: %s", e.Status, e.Msg) }

func (a *App) call(ctx context.Context, base, token, method, path string, body, out any) error {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, base+"/api/v1"+path, rd)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := a.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if resp.StatusCode >= 400 {
		var e struct {
			Error struct{ Message string } `json:"error"`
		}
		_ = json.Unmarshal(b, &e)
		if e.Error.Message == "" {
			e.Error.Message = http.StatusText(resp.StatusCode)
		}
		return &ctlErr{resp.StatusCode, e.Error.Message}
	}
	if out != nil {
		return json.Unmarshal(b, out)
	}
	return nil
}

// verify checks the token and resolves the project.
func (a *App) verify(ctx context.Context, base, token, project string) (*appUser, *appProject, error) {
	var me struct {
		User appUser `json:"user"`
	}
	if err := a.call(ctx, base, token, "GET", "/auth/me", nil, &me); err != nil {
		var ce *ctlErr
		if errors.As(err, &ce) && ce.Status == 401 {
			return nil, nil, errors.New("the controller did not accept this API token (expired or revoked?)")
		}
		return nil, nil, fmt.Errorf("could not reach the controller: %w", err)
	}
	var ps struct {
		Projects []appProject `json:"projects"`
	}
	if err := a.call(ctx, base, token, "GET", "/projects", nil, &ps); err != nil {
		return &me.User, nil, err
	}
	for _, p := range ps.Projects {
		if project == "" || p.Name == project || p.ID == project {
			return &me.User, &p, nil
		}
	}
	if project == "" {
		return &me.User, nil, errors.New("there are no projects on this controller yet")
	}
	return &me.User, nil, fmt.Errorf("project %q not found", project)
}

// snapshot returns the base URL, token and project for a controller call.
func (a *App) snapshot() (base, token, pid string, err error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	base, token = a.controllerURL(), a.cfg.Token
	if base == "" || token == "" {
		return "", "", "", errors.New("set the controller URL and an API token first (Settings)")
	}
	if a.project == nil {
		return base, token, "", errors.New("not connected to the controller: " + a.lastErr)
	}
	return base, token, a.project.ID, nil
}

// refresh re-verifies the stored settings and records the connection state.
func (a *App) refresh(ctx context.Context) {
	a.mu.Lock()
	base, token, project := a.controllerURL(), a.cfg.Token, a.cfg.Project
	a.mu.Unlock()
	var u *appUser
	var p *appProject
	var err error
	if base == "" || token == "" {
		err = errors.New("not configured")
	} else {
		u, p, err = a.verify(ctx, base, token, project)
	}
	a.mu.Lock()
	a.user, a.project, a.connected = u, p, err == nil
	a.lastErr = ""
	if err != nil {
		a.lastErr = err.Error()
	}
	a.mu.Unlock()
}

// ---- background polling ----

func (a *App) poke() {
	select {
	case a.wake <- struct{}{}:
	default:
	}
}

// Run polls unfinished generations until ctx ends.
func (a *App) Run(ctx context.Context) {
	a.refresh(ctx)
	lastRefresh := time.Now()
	for {
		busy := a.pollOnce(ctx)
		if !a.isConnected() && time.Since(lastRefresh) > 30*time.Second {
			a.refresh(ctx)
			lastRefresh = time.Now()
		}
		wait := 15 * time.Second
		if busy {
			wait = 2 * time.Second
		}
		select {
		case <-ctx.Done():
			return
		case <-a.wake:
		case <-time.After(wait):
		}
	}
}

func (a *App) isConnected() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.connected
}

func unfinished(s string) bool { return s == "queued" || s == "running" || s == "downloading" }

type ctlJob struct {
	Status        string `json:"status"`
	FailureReason string `json:"failure_reason"`
	Current       *struct {
		DeviceName string  `json:"device_name"`
		Progress   float64 `json:"progress"`
		Stage      string  `json:"stage"`
	} `json:"current_attempt"`
	Result *struct {
		ArtifactID *string `json:"artifact_id"`
	} `json:"result"`
}

// pollOnce updates every unfinished item; reports whether any remain.
func (a *App) pollOnce(ctx context.Context) bool {
	a.mu.Lock()
	base, token := a.controllerURL(), a.cfg.Token
	var todo []histItem
	for _, it := range a.items {
		if unfinished(it.Status) {
			todo = append(todo, *it)
		}
	}
	a.mu.Unlock()
	if len(todo) == 0 || base == "" || token == "" {
		return len(todo) > 0
	}
	for _, it := range todo {
		if ctx.Err() != nil {
			return true
		}
		var j ctlJob
		err := a.call(ctx, base, token, "GET", "/projects/"+it.ProjectID+"/jobs/"+it.ID, nil, &j)
		a.update(it.ID, func(h *histItem) {
			if err != nil {
				h.pollFailures++
				var ce *ctlErr
				if errors.As(err, &ce) && (ce.Status == 404 || ce.Status == 403) && h.pollFailures >= 3 {
					h.Status, h.Error = "failed", "the controller no longer has this job"
				}
				return
			}
			h.pollFailures = 0
			if j.Current != nil {
				h.Worker, h.Progress, h.Stage = j.Current.DeviceName, j.Current.Progress, j.Current.Stage
			}
			switch j.Status {
			case "completed":
				if j.Result == nil || j.Result.ArtifactID == nil {
					h.Status, h.Error = "failed", "the job finished without a video"
				} else {
					h.Status, h.Progress, h.Stage = "downloading", 1, "downloading"
					h.Params["_artifact"] = *j.Result.ArtifactID
				}
			case "failed", "cancelled":
				h.Status, h.Error = j.Status, j.FailureReason
			case "created", "ready":
				h.Status = "queued"
				if h.Stage == "" {
					h.Stage = "waiting for a free worker"
				}
			default:
				h.Status = "running"
			}
		})
	}
	// Downloads run after the status pass so a large video never delays others.
	a.mu.Lock()
	var dl []histItem
	for _, it := range a.items {
		if it.Status == "downloading" {
			dl = append(dl, *it)
		}
	}
	a.mu.Unlock()
	for _, it := range dl {
		art, _ := it.Params["_artifact"].(string)
		err := a.download(ctx, base, token, "/projects/"+it.ProjectID+"/artifacts/"+art, a.videoPath(it.ID))
		a.update(it.ID, func(h *histItem) {
			if err != nil {
				h.pollFailures++
				if h.pollFailures >= 5 {
					h.Status, h.Error = "failed", "download failed: "+err.Error()
				}
				return
			}
			now := time.Now().UTC()
			h.Status, h.Video, h.FinishedAt, h.Stage, h.pollFailures = "done", true, &now, "", 0
		})
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, it := range a.items {
		if unfinished(it.Status) {
			return true
		}
	}
	return false
}

func (a *App) update(id string, f func(*histItem)) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, it := range a.items {
		if it.ID == id {
			before := *it
			f(it)
			if before.Status != it.Status || before.Progress != it.Progress || before.Stage != it.Stage || before.Worker != it.Worker {
				a.saveHistLocked()
			}
			return
		}
	}
}

func (a *App) download(ctx context.Context, base, token, path, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o700); err != nil {
		return err
	}
	dctx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(dctx, "GET", base+"/api/v1"+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	part := dst + ".part"
	f, err := os.OpenFile(part, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, io.LimitReader(resp.Body, 8<<30)); err != nil {
		f.Close()
		os.Remove(part)
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(part, dst)
}

// ---- HTTP ----

func (a *App) configView() map[string]any {
	a.mu.Lock()
	defer a.mu.Unlock()
	return map[string]any{"controller_url": a.controllerURL(), "has_token": a.cfg.Token != "", "user": a.user,
		"project": a.project, "connected": a.connected, "error": a.lastErr}
}

func (a *App) itemView(h *histItem) map[string]any {
	params := map[string]any{}
	for k, v := range h.Params {
		if !strings.HasPrefix(k, "_") {
			params[k] = v
		}
	}
	var video any
	if h.Video {
		video = "/local/api/app/videos/" + h.ID + ".mp4"
	}
	end := time.Now()
	if h.FinishedAt != nil {
		end = *h.FinishedAt
	}
	return map[string]any{"id": h.ID, "prompt": h.Prompt, "params": params, "created_at": h.CreatedAt, "status": h.Status,
		"progress": h.Progress, "stage": h.Stage, "worker": h.Worker, "error": h.Error, "video_url": video,
		"seconds_total": int(end.Sub(h.CreatedAt).Seconds())}
}

func decodeBody(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16384))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		writeErr(w, 400, "malformed request: "+err.Error())
		return false
	}
	return true
}

// Register adds the app routes to mux.
func (a *App) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /local/api/app/config", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, a.configView()) })
	mux.HandleFunc("PUT /local/api/app/config", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ControllerURL string `json:"controller_url"`
			Token         string `json:"token"`
			Project       string `json:"project"`
		}
		if !decodeBody(w, r, &req) {
			return
		}
		u, err := config.CheckControllerURL(req.ControllerURL, a.AllowInsecure)
		if err != nil {
			writeErr(w, 400, err.Error())
			return
		}
		base := strings.TrimRight(u.String(), "/")
		a.mu.Lock()
		token := a.cfg.Token
		a.mu.Unlock()
		switch tok := strings.TrimSpace(req.Token); tok {
		case "":
		case "-":
			token = ""
		default:
			token = tok
		}
		project := strings.TrimSpace(req.Project)
		if token != "" {
			ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
			defer cancel()
			if _, _, err := a.verify(ctx, base, token, project); err != nil {
				writeErr(w, 400, err.Error())
				return
			}
		}
		a.mu.Lock()
		a.cfg = appConfig{ControllerURL: base, Token: token, Project: project}
		err = a.saveCfgLocked()
		a.mu.Unlock()
		if err != nil {
			writeErr(w, 500, "could not save settings: "+err.Error())
			return
		}
		a.refresh(r.Context())
		a.poke()
		writeJSON(w, 200, a.configView())
	})
	mux.HandleFunc("GET /local/api/app/models", func(w http.ResponseWriter, r *http.Request) {
		base, token, pid, err := a.snapshot()
		if err != nil {
			writeErr(w, 409, err.Error())
			return
		}
		var ms struct {
			Models []struct {
				ID             string  `json:"id"`
				RunName        *string `json:"run_name"`
				Architecture   *string `json:"architecture"`
				Family         string  `json:"family"`
				Active         bool    `json:"active"`
				ParameterCount int64   `json:"parameter_count"`
				Verification   string  `json:"verification_state"`
			} `json:"models"`
		}
		if err := a.call(r.Context(), base, token, "GET", "/projects/"+pid+"/models", nil, &ms); err != nil {
			writeErr(w, 502, err.Error())
			return
		}
		out := []map[string]any{}
		for _, m := range ms.Models {
			name, arch := "", ""
			if m.RunName != nil {
				name = *m.RunName
			}
			if m.Architecture != nil {
				arch = *m.Architecture
			}
			if name == "" {
				name = arch
			}
			fam := m.Family
			if fam != "hd" {
				fam = "small"
			}
			out = append(out, map[string]any{"id": m.ID, "name": name, "architecture": arch, "family": fam, "active": m.Active,
				"parameter_count": m.ParameterCount})
		}
		writeJSON(w, 200, map[string]any{"models": out})
	})
	mux.HandleFunc("POST /local/api/app/generate", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompt       string   `json:"prompt"`
			ModelVersion string   `json:"model_version"`
			Seed         *int64   `json:"seed"`
			Steps        *int     `json:"steps"`
			CfgScale     *float64 `json:"cfg_scale"`
			FPS          *int     `json:"fps"`
			Width        *int     `json:"width"`
			Height       *int     `json:"height"`
			Seconds      *float64 `json:"seconds"`
			Sampler      string   `json:"sampler"`
		}
		if !decodeBody(w, r, &req) {
			return
		}
		req.Prompt = strings.TrimSpace(req.Prompt)
		if req.Prompt == "" || len(req.Prompt) > 2000 {
			writeErr(w, 400, "the prompt must be 1-2000 characters")
			return
		}
		base, token, pid, err := a.snapshot()
		if err != nil {
			writeErr(w, 409, err.Error())
			return
		}
		body := map[string]any{"prompt": req.Prompt}
		if req.ModelVersion != "" {
			body["model_version"] = req.ModelVersion
		}
		seed := time.Now().UnixNano() % (1 << 31)
		if req.Seed != nil {
			seed = *req.Seed
		}
		body["seed"] = seed
		set := func(k string, v any, ok bool) {
			if ok {
				body[k] = v
			}
		}
		set("steps", deref(req.Steps), req.Steps != nil)
		set("cfg_scale", derefF(req.CfgScale), req.CfgScale != nil)
		set("fps", deref(req.FPS), req.FPS != nil)
		set("width", deref(req.Width), req.Width != nil)
		set("height", deref(req.Height), req.Height != nil)
		set("seconds", derefF(req.Seconds), req.Seconds != nil)
		set("sampler", req.Sampler, req.Sampler != "")
		var created struct {
			JobID        string `json:"job_id"`
			ModelVersion string `json:"model_version"`
		}
		if err := a.call(r.Context(), base, token, "POST", "/projects/"+pid+"/generate", body, &created); err != nil {
			status := 502
			var ce *ctlErr
			if errors.As(err, &ce) && ce.Status < 500 {
				status = 400
				err = errors.New(ce.Msg)
			}
			writeErr(w, status, err.Error())
			return
		}
		if !jobIDRe.MatchString(created.JobID) {
			writeErr(w, 502, "the controller returned an unexpected job id")
			return
		}
		params := map[string]any{}
		for k, v := range body {
			if k != "prompt" {
				params[k] = v
			}
		}
		if created.ModelVersion != "" {
			params["model_version"] = created.ModelVersion
		}
		a.mu.Lock()
		a.items = append(a.items, &histItem{ID: created.JobID, ProjectID: pid, Prompt: req.Prompt, Params: params,
			CreatedAt: time.Now().UTC(), Status: "queued", Stage: "waiting for a free worker"})
		a.saveHistLocked()
		a.mu.Unlock()
		a.poke()
		writeJSON(w, 200, map[string]any{"id": created.JobID})
	})
	mux.HandleFunc("GET /local/api/app/history", func(w http.ResponseWriter, r *http.Request) {
		a.mu.Lock()
		out := make([]map[string]any, 0, len(a.items))
		for _, it := range a.items {
			out = append(out, a.itemView(it))
		}
		a.mu.Unlock()
		sort.SliceStable(out, func(i, j int) bool {
			return out[i]["created_at"].(time.Time).After(out[j]["created_at"].(time.Time))
		})
		writeJSON(w, 200, map[string]any{"items": out})
	})
	find := func(w http.ResponseWriter, r *http.Request) (histItem, bool) {
		id := r.PathValue("id")
		if jobIDRe.MatchString(id) {
			a.mu.Lock()
			defer a.mu.Unlock()
			for _, it := range a.items {
				if it.ID == id {
					return *it, true
				}
			}
		}
		writeErr(w, 404, "no such item")
		return histItem{}, false
	}
	mux.HandleFunc("POST /local/api/app/history/{id}/cancel", func(w http.ResponseWriter, r *http.Request) {
		it, ok := find(w, r)
		if !ok {
			return
		}
		if !unfinished(it.Status) {
			writeErr(w, 400, "this generation already finished")
			return
		}
		base, token, _, err := a.snapshot()
		if err != nil {
			writeErr(w, 409, err.Error())
			return
		}
		if err := a.call(r.Context(), base, token, "POST", "/projects/"+it.ProjectID+"/jobs/"+it.ID+"/cancel", nil, nil); err != nil {
			writeErr(w, 502, err.Error())
			return
		}
		a.update(it.ID, func(h *histItem) { h.Status, h.Error = "cancelled", "cancelled by you" })
		writeJSON(w, 200, map[string]any{"ok": true})
	})
	mux.HandleFunc("DELETE /local/api/app/history/{id}", func(w http.ResponseWriter, r *http.Request) {
		it, ok := find(w, r)
		if !ok {
			return
		}
		a.mu.Lock()
		for i, h := range a.items {
			if h.ID == it.ID {
				a.items = append(a.items[:i], a.items[i+1:]...)
				break
			}
		}
		a.saveHistLocked()
		a.mu.Unlock()
		_ = os.Remove(a.videoPath(it.ID))
		writeJSON(w, 200, map[string]any{"ok": true})
	})
	mux.HandleFunc("GET /local/api/app/videos/{file}", func(w http.ResponseWriter, r *http.Request) {
		id := strings.TrimSuffix(r.PathValue("file"), ".mp4")
		if !jobIDRe.MatchString(id) || r.PathValue("file") != id+".mp4" {
			writeErr(w, 404, "no such video")
			return
		}
		f, err := os.Open(a.videoPath(id))
		if err != nil {
			writeErr(w, 404, "no such video")
			return
		}
		defer f.Close()
		st, err := f.Stat()
		if err != nil {
			writeErr(w, 500, "cannot read video")
			return
		}
		w.Header().Set("Content-Type", "video/mp4")
		w.Header().Set("Cache-Control", "private, max-age=3600")
		http.ServeContent(w, r, id+".mp4", st.ModTime(), f)
	})
	mux.HandleFunc("POST /local/api/app/quit", func(w http.ResponseWriter, r *http.Request) {
		if a.Quit == nil {
			writeErr(w, 400, "this worker is not running as the Nivra app")
			return
		}
		writeJSON(w, 200, map[string]any{"ok": true})
		go func() { time.Sleep(200 * time.Millisecond); a.Quit() }()
	})
}

func deref(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func derefF(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}
