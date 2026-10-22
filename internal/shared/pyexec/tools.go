package pyexec

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// Tools are the resolved locations of ffmpeg and ffprobe ("" = not found).
type Tools struct {
	FFmpeg  string
	FFprobe string
}

// OK reports whether both tools were found.
func (t Tools) OK() bool { return t.FFmpeg != "" && t.FFprobe != "" }

func exeName(base string) string {
	if runtime.GOOS == "windows" {
		return base + ".exe"
	}
	return base
}

func isFile(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// ResolveFFmpeg finds ffmpeg and ffprobe. Order: the configured path (a file,
// or a folder holding ffmpeg), then PATH, then the usual winget install folder
// (%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\...\bin), so a
// shortcut started right after installing ffmpeg works even with a stale PATH.
func ResolveFFmpeg(configured string) Tools {
	var t Tools
	configured = strings.TrimSpace(configured)
	if configured != "" {
		if isFile(configured) {
			t.FFmpeg = configured
		} else if p := filepath.Join(configured, exeName("ffmpeg")); isFile(p) {
			t.FFmpeg = p
		} else if p := filepath.Join(configured, "bin", exeName("ffmpeg")); isFile(p) {
			t.FFmpeg = p
		}
	}
	if t.FFmpeg == "" {
		if p, err := exec.LookPath("ffmpeg"); err == nil {
			t.FFmpeg = p
		}
	}
	if t.FFmpeg == "" {
		t.FFmpeg = wingetFind("ffmpeg")
	}
	// ffprobe: next to the chosen ffmpeg first, then PATH, then winget.
	if t.FFmpeg != "" {
		if p := filepath.Join(filepath.Dir(t.FFmpeg), exeName("ffprobe")); isFile(p) {
			t.FFprobe = p
		}
	}
	if t.FFprobe == "" {
		if p, err := exec.LookPath("ffprobe"); err == nil {
			t.FFprobe = p
		}
	}
	if t.FFprobe == "" {
		t.FFprobe = wingetFind("ffprobe")
	}
	return t
}

// wingetFind looks for tool inside winget's Gyan.FFmpeg packages (Windows only).
func wingetFind(tool string) string {
	if runtime.GOOS != "windows" {
		return ""
	}
	la := os.Getenv("LOCALAPPDATA")
	if la == "" {
		return ""
	}
	want := exeName(tool)
	if p := filepath.Join(la, "Microsoft", "WinGet", "Links", want); isFile(p) {
		return p
	}
	pkgs, _ := filepath.Glob(filepath.Join(la, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*"))
	for _, pkg := range pkgs {
		found := ""
		_ = filepath.WalkDir(pkg, func(p string, d os.DirEntry, err error) error {
			if err != nil || found != "" {
				return filepath.SkipDir
			}
			rel, _ := filepath.Rel(pkg, p)
			if d.IsDir() {
				if strings.Count(rel, string(filepath.Separator)) > 3 {
					return filepath.SkipDir
				}
				return nil
			}
			if strings.EqualFold(d.Name(), want) && strings.EqualFold(filepath.Base(filepath.Dir(p)), "bin") {
				found = p
				return filepath.SkipAll
			}
			return nil
		})
		if found != "" {
			return found
		}
	}
	return ""
}

// Env returns the NIVRA_FFMPEG / NIVRA_FFPROBE assignments for a child process.
func (t Tools) Env() []string {
	var env []string
	if t.FFmpeg != "" {
		env = append(env, "NIVRA_FFMPEG="+t.FFmpeg)
	}
	if t.FFprobe != "" {
		env = append(env, "NIVRA_FFPROBE="+t.FFprobe)
	}
	return env
}
