package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"nivra/internal/shared/config"
)

const captionUsage = `Caption videos for training: writes <name>.txt (what is visible) and, when the video has
subtitles or speech, <name>.subtitles.txt next to each video.

  nivra caption PATH [--style auto|anime|"3d animation"|cartoon|"2d animation"|"cel-shaded 3d"|"pixel art"|claymation|painterly|realistic|none]
                     [--subtitles auto|embedded|speech|off] [--frames 6] [--device auto|cpu|cuda]
                     [--overwrite] [--limit N] [--model MODEL_ID_OR_DIR]
  nivra caption --install        Install the captioning dependencies (transformers, faster-whisper)

PATH is a video file or a folder (searched recursively for .mp4 .mov .mkv .webm). Existing .txt files are
skipped unless --overwrite. --limit N captions at most N videos that still need one, so re-running
the same command continues with the next N. Failures are listed in captions-failed.txt. Exit code 0 = all ok, 1 = at least one video failed.
`

// captionPython finds the interpreter and AI package dir the same way generateLocal does,
// plus NIVRA_PYTHON and a repo-local .venv as extra fallbacks.
func captionPython() (python, aiDir string) {
	aiDir = filepath.Join(".", "ai")
	cfgPath := defaultConfig("worker")
	if _, statErr := os.Stat(cfgPath); statErr == nil { // a missing file would yield the built-in default "python"
		if w, err := config.LoadWorker(cfgPath); err == nil && w.PythonPath != "" {
			if w.AIPackageDir != "" {
				aiDir = w.AIPackageDir
			}
			return w.PythonPath, aiDir
		}
	}
	if v := os.Getenv("NIVRA_PYTHON"); v != "" {
		return v, aiDir
	}
	venv := filepath.Join(".venv", "bin", "python")
	if runtime.GOOS == "windows" {
		venv = filepath.Join(".venv", "Scripts", "python.exe")
	}
	if _, err := os.Stat(venv); err == nil {
		return venv, aiDir
	}
	return "python", aiDir
}

func cmdCaption(ctx context.Context, args []string) error {
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" {
		fmt.Print(captionUsage)
		if len(args) == 0 {
			return errors.New("missing PATH")
		}
		return nil
	}
	python, aiDir := captionPython()
	abs, err := filepath.Abs(aiDir)
	if err != nil {
		return err
	}
	var pyArgs []string
	if len(args) == 1 && args[0] == "--install" {
		pyArgs = []string{"-m", "pip", "install", "-r", filepath.Join(abs, "requirements-caption.txt")}
	} else {
		pyArgs = append([]string{"-m", "nivra.cli", "caption"}, args...)
	}
	// exec.Command (not CommandContext): on Ctrl+C the Python process receives the signal itself and stops cleanly.
	cmd := exec.Command(python, pyArgs...)
	_ = ctx
	cmd.Env = append(os.Environ(), "PYTHONPATH="+abs, "PYTHONIOENCODING=utf-8", "PYTHONUTF8=1")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			os.Exit(ee.ExitCode()) // propagate the Python exit code (its output already explained why)
		}
		return fmt.Errorf("could not run %s (is Python installed? run: nivra doctor): %w", python, err)
	}
	return nil
}
