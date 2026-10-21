package localui

import (
	"context"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Model discovery: the app finds Nivra model folders on its own, so a user
// never has to point at one. It searches the usual places (Nivra's models
// folders, Downloads, Desktop, Documents, ComfyUI model folders) a few levels
// deep for a model_version.json, in the background.

const (
	scanDepth    = 5     // levels below each root
	scanMaxDirs  = 40000 // per scan, bounds the time on huge folders
	scanInterval = 2 * time.Minute
)

// skipDir lists folder names that never hold user models and are expensive.
var skipDir = map[string]bool{
	"node_modules": true, "site-packages": true, "__pycache__": true, ".git": true, ".venv": true, "venv": true,
	"appdata": true, "$recycle.bin": true, "windows": true, "program files": true, "program files (x86)": true,
	"system volume information": true, "go": true, "pkg": true, "cache": true, ".cache": true,
}

// DefaultScanRoots returns the folders searched for models on this computer.
func DefaultScanRoots() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	var out []string
	for _, d := range []string{"Downloads", "Desktop", "Documents", "Videos", "Nivra", "ComfyUI",
		filepath.Join("OneDrive", "Desktop"), filepath.Join("OneDrive", "Documents")} {
		out = append(out, filepath.Join(home, d))
	}
	return out
}

// scanForModels walks roots and returns every folder holding a Nivra manifest.
func scanForModels(ctx context.Context, roots []string) []string {
	var found []string
	seen := map[string]bool{}
	visited := 0
	for _, root := range roots {
		root = filepath.Clean(root)
		if st, err := os.Stat(root); err != nil || !st.IsDir() {
			continue
		}
		base := strings.Count(root, string(filepath.Separator))
		_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil || ctx.Err() != nil {
				if d != nil && d.IsDir() {
					return fs.SkipDir
				}
				return nil
			}
			if !d.IsDir() {
				return nil
			}
			if visited++; visited > scanMaxDirs {
				return fs.SkipAll
			}
			name := strings.ToLower(d.Name())
			if p != root && (strings.HasPrefix(name, ".") || skipDir[name]) {
				return fs.SkipDir
			}
			if d.Type()&fs.ModeSymlink != 0 {
				return fs.SkipDir
			}
			if _, err := os.Stat(filepath.Join(p, "model_version.json")); err == nil {
				if _, err := inspectBundle(p); err == nil && !seen[strings.ToLower(p)] {
					seen[strings.ToLower(p)] = true
					found = append(found, p)
				}
				return fs.SkipDir // a model folder holds no further models
			}
			if strings.Count(p, string(filepath.Separator))-base >= scanDepth {
				return fs.SkipDir
			}
			return nil
		})
	}
	return found
}

// rescan refreshes the discovered model list.
func (a *App) rescan(ctx context.Context) {
	if len(a.ScanRoots) == 0 {
		return
	}
	found := scanForModels(ctx, a.ScanRoots)
	a.mu.Lock()
	a.discovered = found
	a.mu.Unlock()
}

// runDiscovery rescans now and then every few minutes.
func (a *App) runDiscovery(ctx context.Context) {
	for {
		a.rescan(ctx)
		select {
		case <-ctx.Done():
			return
		case <-a.rescanNow:
		case <-time.After(scanInterval):
		}
	}
}

func (a *App) registerDiscovery(mux *http.ServeMux) {
	mux.HandleFunc("POST /local/api/app/local-models/rescan", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
		defer cancel()
		a.rescan(ctx)
		writeJSON(w, 200, map[string]any{"models": len(a.localModels())})
	})
}
