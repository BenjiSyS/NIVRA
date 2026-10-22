package localui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"nivra/internal/shared/config"
	"nivra/internal/worker/agent"
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
	// Runtime, when set, lets the app run model bundles on this computer.
	Runtime func() LocalRuntime
	// ModelDirs are scanned for model bundle folders (each subfolder one model).
	ModelDirs []string
	// ScanRoots are searched a few levels deep, in the background, for models.
	ScanRoots []string
	// Gate, when set, keeps local generations and network tasks off the GPU at the
	// same time (shared with the worker agents).
	Gate *agent.Gate
	// KeepAwake, when set, blocks computer sleep until the returned function is called.
	KeepAwake func() func()

	mu        sync.Mutex
	cfg       appConfig
	items     []*histItem
	user      *appUser
	project   *appProject
	connected bool
	lastErr   string
	hc        *http.Client
	wake      chan struct{}
	// local generation queue
	localWake   chan struct{}
	localCancel map[string]context.CancelFunc
	discovered  []string
	rescanNow   chan struct{}
}

type appConfig struct {
	ControllerURL string `json:"controller_url"`
	Token         string `json:"token"`
	Project       string `json:"project"`
	// LocalModels are model bundle folders loaded with "Load model".
	LocalModels []string `json:"local_models,omitempty"`
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
	ID         string         `json:"id"`
	ProjectID  string         `json:"project_id"`
	Prompt     string         `json:"prompt"`
	Params     map[string]any `json:"params"`
	CreatedAt  time.Time      `json:"created_at"`
	FinishedAt *time.Time     `json:"finished_at,omitempty"`
	Status     string         `json:"status"`
	Progress   float64        `json:"progress"`
	Stage      string         `json:"stage"`
	Worker     string         `json:"worker"`
	Error      string         `json:"error"`
	Video      bool           `json:"video"`
	// Local generations run on this computer from a loaded model folder.
	Local bool `json:"local,omitempty"`
	// GenerationID is set for long videos built from segments on the network.
	GenerationID string `json:"generation_id,omitempty"`
	// WaitingReason says why nothing can run the job yet (from the controller; may be empty).
	WaitingReason string     `json:"waiting_reason,omitempty"`
	ModelID       string     `json:"model_id,omitempty"`
	StartedAt     *time.Time `json:"started_at,omitempty"`
	pollFailures  int
	samples       []progSample
}

// progSample is one (time, progress) observation for the time-left estimate.
type progSample struct {
	t time.Time
	p float64
}

// eta estimates the seconds left from the recent progress rate; -1 = unknown.
func (h *histItem) eta(now time.Time) int {
	if !unfinished(h.Status) || h.Status == "queued" || len(h.samples) < 2 {
		return -1
	}
	first, last := h.samples[0], h.samples[len(h.samples)-1]
	dp := last.p - first.p
	dt := last.t.Sub(first.t).Seconds()
	if dp < 0.005 || dt < 1 || last.p >= 1 {
		return -1
	}
	left := (1 - last.p) * dt / dp
	// Time since the last update already counts towards what is left.
	left -= now.Sub(last.t).Seconds()
	if left < 1 {
		left = 1
	}
	return int(left + 0.5)
}

// internalParams are history params that are not generation parameters.
var internalParams = map[string]bool{"segments": true}

// maxPromptRunes is the longest prompt the app accepts (HD models); small models take 200.
const (
	maxPromptRunes      = 1000
	maxSmallPromptRunes = 200
)

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
	a.localWake = make(chan struct{}, 1)
	a.rescanNow = make(chan struct{}, 1)
	a.localCancel = map[string]context.CancelFunc{}
	for _, it := range a.items {
		// A local generation dies with the app; network jobs keep running.
		if it.Local && (it.Status == "running" || it.Status == "downloading") {
			it.Status, it.Error = "failed", "interrupted: Nivra was closed while this was generating"
		}
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

func (e *ctlErr) Error() string { return fmt.Sprintf("Nivra server: HTTP %d: %s", e.Status, e.Msg) }

// friendlyErr turns a raw network or server error into a short sentence for people.
func friendlyErr(err error) string {
	var ce *ctlErr
	if errors.As(err, &ce) {
		switch {
		case ce.Status == 401:
			return "That username or password is not right."
		case ce.Status == 403:
			return "The Nivra server says you are not allowed to do that: " + ce.Msg
		case ce.Status == 429:
			return "Too many tries. Wait a minute and try again."
		case ce.Status >= 500:
			return "The Nivra server had a problem. Try again in a minute."
		}
		return ce.Msg
	}
	var ne net.Error
	msg := strings.ToLower(err.Error())
	switch {
	case errors.Is(err, context.DeadlineExceeded), errors.As(err, &ne) && ne.Timeout():
		return "The Nivra server did not answer in time. Check the address and your internet connection."
	case strings.Contains(msg, "no such host"), strings.Contains(msg, "connection refused"),
		strings.Contains(msg, "dial tcp"), strings.Contains(msg, "tls:"), strings.Contains(msg, "certificate"):
		return "Cannot reach the Nivra server at that address. Check the address and your internet connection."
	}
	return "Could not talk to the Nivra server: " + err.Error()
}

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
			return nil, nil, errors.New("the Nivra server did not accept this access key (it may have expired or been removed)")
		}
		return nil, nil, errors.New(friendlyErr(err))
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
		return &me.User, nil, errors.New("there are no projects on this Nivra server yet")
	}
	return &me.User, nil, fmt.Errorf("project %q not found on the Nivra server", project)
}

// snapshot returns the base URL, token and project for a controller call.
func (a *App) snapshot() (base, token, pid string, err error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	base, token = a.controllerURL(), a.cfg.Token
	if base == "" || token == "" {
		return "", "", "", errors.New("connect to a Nivra server first (Connection)")
	}
	if a.project == nil {
		return base, token, "", errors.New("not connected to the Nivra server: " + a.lastErr)
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
	go a.runLocal(ctx)
	go a.runDiscovery(ctx)
	a.pokeLocal()
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

func (a *App) pokeLocal() {
	select {
	case a.localWake <- struct{}{}:
	default:
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
	WaitingReason string `json:"waiting_reason"`
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
		if unfinished(it.Status) && !it.Local {
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
		if it.GenerationID != "" {
			a.pollGeneration(ctx, base, token, it)
			continue
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
				h.WaitingReason = j.WaitingReason
				if h.Stage == "" {
					h.Stage = "waiting for a free worker"
				}
			default:
				h.Status = "running"
				h.WaitingReason = ""
			}
		})
	}
	// Downloads run after the status pass so a large video never delays others.
	a.mu.Lock()
	var dl []histItem
	for _, it := range a.items {
		if it.Status == "downloading" && !it.Local {
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
		if unfinished(it.Status) && !it.Local {
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
			if it.Progress != before.Progress || (it.Status == "running" && before.Status != "running") {
				it.samples = append(it.samples, progSample{time.Now(), it.Progress})
				// Keep the last ~3 minutes so the rate follows the current stage.
				for len(it.samples) > 2 && time.Since(it.samples[0].t) > 3*time.Minute {
					it.samples = it.samples[1:]
				}
			}
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
		// "_..." keys and "segments" are the app's own bookkeeping; the controller
		// rejects them as unknown fields when Retry posts the params back.
		if !strings.HasPrefix(k, "_") && !internalParams[k] {
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
	var eta any
	if e := h.eta(time.Now()); e >= 0 {
		eta = e
	}
	return map[string]any{"id": h.ID, "prompt": h.Prompt, "params": params, "created_at": h.CreatedAt, "status": h.Status,
		"progress": h.Progress, "stage": h.Stage, "worker": h.Worker, "error": h.Error, "video_url": video,
		"seconds_total": int(end.Sub(h.CreatedAt).Seconds()), "eta_seconds": eta, "local": h.Local,
		"waiting_reason": h.WaitingReason}
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
		a.cfg.ControllerURL, a.cfg.Token, a.cfg.Project = base, token, project
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
	a.registerLogin(mux)
	a.registerLocal(mux)
	a.registerDiscovery(mux)
	mux.HandleFunc("GET /local/api/app/models", func(w http.ResponseWriter, r *http.Request) {
		out := []map[string]any{}
		for _, m := range a.localModels() {
			out = append(out, map[string]any{"id": m.ID, "name": m.Name, "architecture": "", "family": m.Family, "active": false,
				"parameter_count": m.Params, "styles": m.Styles, "default_style": m.DefaultStyle, "source": "local", "path": m.Path,
				"default_seconds": m.DefaultSeconds, "default_fps": m.DefaultFPS})
		}
		base, token, pid, err := a.snapshot()
		if err != nil {
			// No controller: local models still work.
			writeJSON(w, 200, map[string]any{"models": out, "network_error": err.Error()})
			return
		}
		var ms struct {
			Models []struct {
				ID             string   `json:"id"`
				RunName        *string  `json:"run_name"`
				Architecture   *string  `json:"architecture"`
				Family         string   `json:"family"`
				Active         bool     `json:"active"`
				ParameterCount int64    `json:"parameter_count"`
				Verification   string   `json:"verification_state"`
				Styles         []string `json:"styles"`
				DefaultStyle   string   `json:"default_style"`
			} `json:"models"`
		}
		if err := a.call(r.Context(), base, token, "GET", "/projects/"+pid+"/models", nil, &ms); err != nil {
			writeJSON(w, 200, map[string]any{"models": out, "network_error": err.Error()})
			return
		}
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
			styles := m.Styles
			if styles == nil {
				styles = []string{}
			}
			out = append(out, map[string]any{"id": m.ID, "name": name, "architecture": arch, "family": fam, "active": m.Active,
				"parameter_count": m.ParameterCount, "styles": styles, "default_style": m.DefaultStyle, "source": "network"})
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
			Style        string   `json:"style"`
		}
		if !decodeBody(w, r, &req) {
			return
		}
		req.Prompt = strings.TrimSpace(req.Prompt)
		if n := utf8.RuneCountInString(req.Prompt); n == 0 || n > maxPromptRunes {
			writeErr(w, 400, "the prompt must be 1-1000 characters")
			return
		}
		local := strings.HasPrefix(req.ModelVersion, localPrefix)
		var base, token, pid string
		if local {
			if a.Runtime == nil {
				writeErr(w, 400, "this app cannot run models locally")
				return
			}
			lm, ok := a.findLocal(req.ModelVersion)
			if !ok {
				writeErr(w, 400, "that model folder is no longer available; load it again")
				return
			}
			if lm.Family != "hd" && utf8.RuneCountInString(req.Prompt) > maxSmallPromptRunes {
				writeErr(w, 400, "this model takes prompts of 1-200 characters")
				return
			}
			if err := a.Gate.NetworkBusy(); err != nil {
				writeErr(w, 409, err.Error())
				return
			}
		} else {
			var err error
			if base, token, pid, err = a.snapshot(); err != nil {
				writeErr(w, 409, err.Error())
				return
			}
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
		set("style", req.Style, req.Style != "")
		if local {
			params := map[string]any{}
			for k, v := range body {
				if k != "prompt" {
					params[k] = v
				}
			}
			id := newLocalJobID()
			a.mu.Lock()
			a.items = append(a.items, &histItem{ID: id, Prompt: req.Prompt, Params: params, CreatedAt: time.Now().UTC(),
				Status: "queued", Stage: "waiting for this computer", Local: true, ModelID: req.ModelVersion})
			a.saveHistLocked()
			a.mu.Unlock()
			a.pokeLocal()
			writeJSON(w, 200, map[string]any{"id": id})
			return
		}
		var created struct {
			JobID        string `json:"job_id"`
			GenerationID string `json:"generation_id"`
			ModelVersion string `json:"model_version"`
			Segments     int    `json:"segments"`
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
		if !jobIDRe.MatchString(created.JobID) || (created.GenerationID != "" && !jobIDRe.MatchString(created.GenerationID)) {
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
		item := &histItem{ID: created.JobID, ProjectID: pid, Prompt: req.Prompt, Params: params,
			CreatedAt: time.Now().UTC(), Status: "queued", Stage: "waiting for a free worker"}
		if created.GenerationID != "" {
			// Long video: the history entry follows the whole generation.
			item.ID, item.GenerationID = created.GenerationID, created.GenerationID
			item.Params["segments"] = created.Segments
		}
		a.items = append(a.items, item)
		a.saveHistLocked()
		a.mu.Unlock()
		a.poke()
		writeJSON(w, 200, map[string]any{"id": item.ID})
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
		if it.Local {
			a.update(it.ID, func(h *histItem) { h.Status, h.Error = "cancelled", "cancelled by you" })
			a.mu.Lock()
			if c := a.localCancel[it.ID]; c != nil {
				c()
			}
			a.mu.Unlock()
			writeJSON(w, 200, map[string]any{"ok": true})
			return
		}
		base, token, _, err := a.snapshot()
		if err != nil {
			writeErr(w, 409, err.Error())
			return
		}
		path := "/projects/" + it.ProjectID + "/jobs/" + it.ID + "/cancel"
		if it.GenerationID != "" {
			path = "/projects/" + it.ProjectID + "/generations/" + it.GenerationID + "/cancel"
		}
		if err := a.call(r.Context(), base, token, "POST", path, nil, nil); err != nil {
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

// pollGeneration updates a long-video item from the controller's generation view.
func (a *App) pollGeneration(ctx context.Context, base, token string, it histItem) {
	var g struct {
		Status        string   `json:"status"`
		Stage         string   `json:"stage"`
		Progress      float64  `json:"progress"`
		SegmentsTotal int      `json:"segments_total"`
		SegmentsDone  int      `json:"segments_done"`
		Workers       []string `json:"workers"`
		ArtifactID    *string  `json:"artifact_id"`
		Failure       string   `json:"failure_reason"`
		WaitingReason string   `json:"waiting_reason"`
	}
	err := a.call(ctx, base, token, "GET", "/projects/"+it.ProjectID+"/generations/"+it.GenerationID, nil, &g)
	a.update(it.ID, func(h *histItem) {
		if err != nil {
			h.pollFailures++
			var ce *ctlErr
			if errors.As(err, &ce) && (ce.Status == 404 || ce.Status == 403) && h.pollFailures >= 3 {
				h.Status, h.Error = "failed", "the controller no longer has this video"
			}
			return
		}
		h.pollFailures = 0
		h.Progress = g.Progress
		h.Worker = strings.Join(g.Workers, ", ")
		h.WaitingReason = ""
		if g.Progress == 0 && len(g.Workers) == 0 {
			h.WaitingReason = g.WaitingReason
		}
		switch g.Stage {
		case "draft":
			h.Stage = "drawing the whole video at low resolution"
		case "refine":
			h.Stage = fmt.Sprintf("rendering segments: %d of %d done", g.SegmentsDone, g.SegmentsTotal)
			if n := len(g.Workers); n > 1 {
				h.Stage += fmt.Sprintf(" on %d computers", n)
			}
		case "stitch":
			h.Stage = "joining the segments"
		}
		switch g.Status {
		case "completed":
			if g.ArtifactID == nil {
				h.Status, h.Error = "failed", "the video finished without a file"
				return
			}
			h.Status, h.Progress, h.Stage = "downloading", 1, "downloading"
			h.Params["_artifact"] = *g.ArtifactID
		case "failed", "cancelled":
			h.Status, h.Error = g.Status, g.Failure
		default:
			if g.Progress > 0 || len(g.Workers) > 0 {
				h.Status = "running"
			}
		}
	})
}
