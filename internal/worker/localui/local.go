package localui

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"nivra/internal/shared/procattr"
	"nivra/internal/shared/pyexec"
)

// LocalRuntime says how to run the Python AI executor on this computer.
type LocalRuntime struct {
	Python  string
	AIDir   string
	Device  string // cpu | cuda:N
	Threads int
	// Devices are every local device that may render long-video segments
	// at once (each CUDA GPU of the worker slots; just Device on CPU).
	Devices []string
}

// localModel is a model bundle folder on this computer.
type localModel struct {
	ID           string   `json:"id"` // "local:" + hash of the path
	Path         string   `json:"path"`
	Name         string   `json:"name"`
	Family       string   `json:"family"`
	Params       int64    `json:"parameter_count"`
	Styles       []string `json:"styles"`
	DefaultStyle string   `json:"default_style"`
	// One model clip; longer videos are built from segments.
	DefaultSeconds float64 `json:"default_seconds"`
	DefaultFPS     int     `json:"default_fps"`
}

const localPrefix = "local:"

func localID(path string) string {
	sum := sha256.Sum256([]byte(strings.ToLower(filepath.Clean(path))))
	return localPrefix + hex.EncodeToString(sum[:6])
}

// inspectBundle validates a bundle folder and reads what the app shows.
func inspectBundle(dir string) (localModel, error) {
	dir = filepath.Clean(strings.TrimSpace(dir))
	if !filepath.IsAbs(dir) {
		return localModel{}, errors.New("give the full path of the model folder")
	}
	b, err := os.ReadFile(filepath.Join(dir, "model_version.json"))
	if err != nil {
		return localModel{}, errors.New("this folder is not a Nivra model: model_version.json is missing")
	}
	var m struct {
		Schema       string   `json:"schema"`
		ConfigName   string   `json:"config_name"`
		Family       string   `json:"family"`
		Params       float64  `json:"param_count"`
		Styles       []string `json:"styles"`
		DefaultStyle string   `json:"default_style"`
		DefSeconds   float64  `json:"default_seconds"`
		DefFPS       float64  `json:"default_fps"`
	}
	if err := json.Unmarshal(b, &m); err != nil || !strings.HasPrefix(m.Schema, "nivra.model_version/") {
		return localModel{}, errors.New("model_version.json is not a Nivra model manifest")
	}
	for _, f := range []string{"weights.safetensors", "vae.safetensors"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			return localModel{}, fmt.Errorf("the model folder is incomplete: %s is missing", f)
		}
	}
	fam := "small"
	if m.Family == "hd" {
		fam = "hd"
	}
	if m.Styles == nil {
		m.Styles = []string{}
	}
	name := filepath.Base(dir)
	if m.ConfigName != "" && m.ConfigName != name {
		name += " (" + m.ConfigName + ")"
	}
	if m.DefSeconds <= 0 {
		m.DefSeconds = 8
	}
	if m.DefFPS <= 0 {
		m.DefFPS = 30
	}
	return localModel{ID: localID(dir), Path: dir, Name: name, Family: fam, Params: int64(m.Params),
		Styles: m.Styles, DefaultStyle: m.DefaultStyle, DefaultSeconds: m.DefSeconds, DefaultFPS: int(m.DefFPS)}, nil
}

// localModels returns the loaded bundles plus bundles found in the default
// model folders (<install>/models and <data_dir>/models).
func (a *App) localModels() []localModel {
	a.mu.Lock()
	paths := append(append([]string(nil), a.cfg.LocalModels...), a.discovered...)
	a.mu.Unlock()
	for _, root := range a.ModelDirs {
		ents, _ := os.ReadDir(root)
		for _, e := range ents {
			if e.IsDir() {
				paths = append(paths, filepath.Join(root, e.Name()))
			}
		}
	}
	seen := map[string]bool{}
	out := []localModel{}
	for _, p := range paths {
		m, err := inspectBundle(p)
		if err != nil || seen[m.ID] {
			continue
		}
		seen[m.ID] = true
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (a *App) findLocal(id string) (localModel, bool) {
	for _, m := range a.localModels() {
		if m.ID == id {
			return m, true
		}
	}
	return localModel{}, false
}

func newLocalJobID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	h := hex.EncodeToString(b[:])
	return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:32]
}

// browseFolder shows the Windows folder picker on this computer.
func browseFolder(ctx context.Context) (string, error) {
	if runtime.GOOS != "windows" {
		return "", errors.New("the folder picker is only available on Windows; paste the folder path instead")
	}
	script := `Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.FolderBrowserDialog
$d.Description = 'Choose a Nivra model folder (it contains model_version.json)'
$d.ShowNewFolderButton = $false
$owner = New-Object System.Windows.Forms.Form -Property @{TopMost = $true}
if ($d.ShowDialog($owner) -eq 'OK') { [Console]::Out.Write($d.SelectedPath) }`
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-STA", "-NonInteractive", "-Command", script)
	procattr.HideConsole(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("folder picker failed: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}

// ---- local generation ----

type localReporter struct {
	a  *App
	id string
}

func (r localReporter) Progress(f float64, stage string) {
	r.a.update(r.id, func(h *histItem) {
		if h.Status == "running" { // never revive a cancelled item
			h.Progress, h.Stage = f, stage
		}
	})
}
func (r localReporter) Log(string, string)            {}
func (r localReporter) Metric(string, float64, int64) {}

// runLocal works through queued local generations one at a time.
func (a *App) runLocal(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-a.localWake:
		}
		for {
			it, ok := a.nextLocal()
			if !ok {
				break
			}
			a.generateLocal(ctx, it)
		}
	}
}

func (a *App) nextLocal() (histItem, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	var next *histItem
	for _, it := range a.items {
		if it.Local && it.Status == "queued" && (next == nil || it.CreatedAt.Before(next.CreatedAt)) {
			next = it
		}
	}
	if next == nil {
		return histItem{}, false
	}
	return *next, true
}

func (a *App) generateLocal(ctx context.Context, it histItem) {
	fail := func(msg string) {
		a.update(it.ID, func(h *histItem) { h.Status, h.Error = "failed", msg })
	}
	m, ok := a.findLocal(it.ModelID)
	if !ok {
		fail("the model folder is no longer available")
		return
	}
	if a.Runtime == nil {
		fail("this app cannot run models locally")
		return
	}
	rt := a.Runtime()
	if rt.Python == "" {
		fail("Python/PyTorch is not set up on this computer; run Nivra-Setup.bat again")
		return
	}
	jctx, cancel := context.WithCancel(ctx)
	a.mu.Lock()
	a.localCancel[it.ID] = cancel
	a.mu.Unlock()
	defer func() {
		cancel()
		a.mu.Lock()
		delete(a.localCancel, it.ID)
		a.mu.Unlock()
	}()
	started := false
	a.update(it.ID, func(h *histItem) {
		if h.Status != "queued" {
			return // cancelled while waiting
		}
		started = true
		h.Status, h.Stage, h.Worker = "running", "loading the model", "this computer"
		now := time.Now().UTC()
		h.StartedAt = &now
	})
	if !started {
		return
	}
	work := filepath.Join(a.Dir, "app-local", it.ID)
	_ = os.RemoveAll(work)
	if err := os.MkdirAll(work, 0o700); err != nil {
		fail(err.Error())
		return
	}
	defer os.RemoveAll(work)
	if secs, _ := it.Params["seconds"].(float64); m.Family == "hd" && secs > m.DefaultSeconds {
		video, err := a.generateLocalLong(jctx, it, m, rt, work)
		if jctx.Err() != nil {
			a.update(it.ID, func(h *histItem) {
				if h.Status != "cancelled" {
					h.Status, h.Error = "cancelled", "stopped"
				}
			})
			return
		}
		if err != nil {
			fail(err.Error())
			return
		}
		a.finishLocal(it.ID, video, fail)
		return
	}
	params := map[string]any{"prompt": it.Prompt, "name": "nivra-local"}
	for k, v := range it.Params {
		if k != "model_version" && !strings.HasPrefix(k, "_") {
			params[k] = v
		}
	}
	pb, _ := json.Marshal(params)
	run := &pyexec.Runner{Interpreter: rt.Python, AIDir: rt.AIDir, Threads: rt.Threads, Device: rt.Device}
	res, err := run.Run(jctx, work, pyexec.Job{JobType: "ai.generate", JobID: it.ID, AttemptID: it.ID, Params: pb,
		Inputs: map[string]any{"weights": filepath.Join(m.Path, "weights.safetensors"),
			"manifest": filepath.Join(m.Path, "model_version.json"), "vae_weights": filepath.Join(m.Path, "vae.safetensors")}},
		localReporter{a, it.ID})
	if errors.Is(err, pyexec.ErrStopped) || jctx.Err() != nil {
		a.update(it.ID, func(h *histItem) {
			if h.Status != "cancelled" {
				h.Status, h.Error = "cancelled", "stopped"
			}
		})
		return
	}
	var f *pyexec.Failure
	if errors.As(err, &f) {
		fail(f.Message)
		return
	}
	if err != nil {
		fail(err.Error())
		return
	}
	src := ""
	for _, art := range res.Artifacts {
		if strings.HasSuffix(art.Path, ".mp4") {
			src = art.Path
		}
	}
	if src == "" {
		fail("the generation finished without a video")
		return
	}
	if !filepath.IsAbs(src) {
		src = filepath.Join(work, src)
	}
	a.finishLocal(it.ID, src, fail)
}

// finishLocal saves the finished video and marks the item done.
func (a *App) finishLocal(id, src string, fail func(string)) {
	if err := os.MkdirAll(a.videoDir(), 0o700); err != nil {
		fail(err.Error())
		return
	}
	if err := moveFile(src, a.videoPath(id)); err != nil {
		fail("could not save the video: " + err.Error())
		return
	}
	a.update(id, func(h *histItem) {
		if h.Status != "running" {
			return
		}
		now := time.Now().UTC()
		h.Status, h.Video, h.FinishedAt, h.Stage, h.Progress = "done", true, &now, "", 1
	})
}

func moveFile(src, dst string) error {
	if err := os.Rename(src, dst); err == nil {
		return nil
	}
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, b, 0o600)
}

// registerLocal adds the local-model routes.
func (a *App) registerLocal(mux *http.ServeMux) {
	add := func(w http.ResponseWriter, path string) {
		m, err := inspectBundle(path)
		if err != nil {
			writeErr(w, 400, err.Error())
			return
		}
		a.mu.Lock()
		dup := false
		for _, p := range a.cfg.LocalModels {
			if localID(p) == m.ID {
				dup = true
			}
		}
		if !dup {
			a.cfg.LocalModels = append(a.cfg.LocalModels, m.Path)
		}
		err = a.saveCfgLocked()
		a.mu.Unlock()
		if err != nil {
			writeErr(w, 500, "could not save settings: "+err.Error())
			return
		}
		writeJSON(w, 200, m)
	}
	mux.HandleFunc("POST /local/api/app/local-models", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Path string `json:"path"`
		}
		if !decodeBody(w, r, &req) {
			return
		}
		add(w, req.Path)
	})
	mux.HandleFunc("POST /local/api/app/local-models/browse", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
		defer cancel()
		p, err := browseFolder(ctx)
		if err != nil {
			writeErr(w, 400, err.Error())
			return
		}
		if p == "" {
			writeJSON(w, 200, map[string]any{"cancelled": true})
			return
		}
		add(w, p)
	})
	mux.HandleFunc("DELETE /local/api/app/local-models/{id}", func(w http.ResponseWriter, r *http.Request) {
		id := localPrefix + strings.TrimPrefix(r.PathValue("id"), localPrefix)
		a.mu.Lock()
		kept := a.cfg.LocalModels[:0]
		for _, p := range a.cfg.LocalModels {
			if localID(p) != id {
				kept = append(kept, p)
			}
		}
		a.cfg.LocalModels = kept
		err := a.saveCfgLocked()
		a.mu.Unlock()
		if err != nil {
			writeErr(w, 500, err.Error())
			return
		}
		writeJSON(w, 200, map[string]any{"ok": true})
	})
}
