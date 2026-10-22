package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/term"

	"nivra/internal/shared/config"
)

const generateUsage = `Generate a video with a Nivra model.

Remote (any computer; the work runs on a connected worker computer):
  nivra generate --url https://studio.example.com --user alice --prompt "a red circle moves from left to right" --out clip.mp4
  Password: prompted, or from the NIVRA_PASSWORD environment variable. The account needs the member role or higher (invite a friend from Studio, People).

Local (one computer, no controller; needs Python + PyTorch from the worker install):
  nivra generate --bundle path/to/model-bundle --prompt "two blue squares move up" --out clip.mp4

Options:
`

func cmdGenerate(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	fs.Usage = func() { fmt.Fprint(os.Stderr, generateUsage); fs.PrintDefaults() }
	url := fs.String("url", "", "controller URL (remote mode)")
	user := fs.String("user", "", "account name (remote mode)")
	project := fs.String("project", "", "project name (remote mode; default: the first project)")
	model := fs.String("model", "", "model version id (default: the project's active model)")
	bundle := fs.String("bundle", "", "local model bundle directory (local mode)")
	prompt := fs.String("prompt", "", "what to generate (English or Swedish)")
	seed := fs.Int("seed", 0, "random seed")
	steps := fs.Int("steps", 30, "sampling steps")
	cfg := fs.Float64("cfg", 3.0, "guidance (CFG) scale")
	fps := fs.Int("fps", 8, "frames per second of the MP4")
	upscale := fs.Int("upscale", 4, "nearest-neighbour upscale factor for viewing")
	out := fs.String("out", "nivra.mp4", "output MP4 path")
	width := fs.Int("width", 0, "HD models: output width (default from the model, e.g. 1280)")
	height := fs.Int("height", 0, "HD models: output height (default from the model, e.g. 720)")
	seconds := fs.Float64("seconds", 0, "HD models: duration in seconds (default from the model, e.g. 8)")
	sampler := fs.String("sampler", "", "HD models: euler or ab2")
	style := fs.String("style", "", `HD models: style tag ("anime", "3d animation", "cartoon", ...; "none" = no tag; default from the model)`)
	python := fs.String("python", "", "Python interpreter (local mode; default from config/worker.json)")
	_ = fs.Parse(args)
	if strings.TrimSpace(*prompt) == "" {
		fs.Usage()
		return errors.New("--prompt is required")
	}
	switch {
	case *bundle != "":
		return generateLocal(ctx, *bundle, *python, *prompt, *seed, *steps, *cfg, *fps, *upscale, *out)
	case *url != "":
		hd = map[string]any{}
		if *width > 0 {
			hd["width"] = *width
		}
		if *height > 0 {
			hd["height"] = *height
		}
		if *seconds > 0 {
			hd["seconds"] = *seconds
		}
		if *sampler != "" {
			hd["sampler"] = *sampler
		}
		if *style != "" {
			hd["style"] = *style
		}
		return generateRemote(ctx, *url, *user, *project, *model, *prompt, *seed, *steps, *cfg, *fps, *upscale, *out)
	default:
		fs.Usage()
		return errors.New("use --url (remote) or --bundle (local)")
	}
}

func generateLocal(ctx context.Context, bundle, python, prompt string, seed, steps int, cfg float64, fps, upscale int, out string) error {
	aiDir := filepath.Join(".", "ai")
	if python == "" {
		if w, err := config.LoadWorker(defaultConfig("worker")); err == nil && w.PythonPath != "" {
			python = w.PythonPath
			if w.AIPackageDir != "" {
				aiDir = w.AIPackageDir
			}
		} else {
			python = "python"
		}
	}
	tmp, err := os.MkdirTemp("", "nivra-gen-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	cmd := exec.CommandContext(ctx, python, "-m", "nivra.cli", "generate", "--bundle", bundle, "--prompt", prompt, "--out", tmp,
		"--name", "video", "--seed", fmt.Sprint(seed), "--steps", fmt.Sprint(steps), "--cfg", fmt.Sprint(cfg),
		"--fps", fmt.Sprint(fps), "--upscale", fmt.Sprint(upscale))
	abs, _ := filepath.Abs(aiDir)
	cmd.Env = append(os.Environ(), "PYTHONPATH="+abs, "PYTHONIOENCODING=utf-8", "PYTHONUTF8=1")
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	fmt.Fprintf(os.Stderr, "generating locally with %s ...\n", bundle)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("local generation failed (is the AI executor installed? run: nivra doctor): %w", err)
	}
	if err := copyFile(filepath.Join(tmp, "video.mp4"), out); err != nil {
		return err
	}
	if b, err := os.ReadFile(filepath.Join(tmp, "video.json")); err == nil {
		_ = os.WriteFile(strings.TrimSuffix(out, filepath.Ext(out))+".json", b, 0o644)
	}
	fmt.Printf("saved %s\n", out)
	return nil
}

// hd holds optional HD-model parameters (width, height, seconds, sampler).
var hd map[string]any

type remote struct {
	base  string
	hc    *http.Client
	csrf  string
	token string
}

func (r *remote) do(ctx context.Context, method, path string, body, out any) error {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, r.base+path, rd)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if r.csrf != "" {
		req.Header.Set("X-Nivra-CSRF", r.csrf)
	}
	if r.token != "" {
		req.Header.Set("Authorization", "Bearer "+r.token)
	}
	resp, err := r.hc.Do(req)
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
		return fmt.Errorf("%s %s: HTTP %d %s", method, path, resp.StatusCode, e.Error.Message)
	}
	if out != nil {
		return json.Unmarshal(b, out)
	}
	return nil
}

func generateRemote(ctx context.Context, url, user, project, model, prompt string, seed, steps int, cfg float64, fps, upscale int, out string) error {
	u, err := config.CheckControllerURL(url, false)
	if err != nil {
		return err
	}
	jar, _ := cookiejar.New(nil)
	r := &remote{base: strings.TrimRight(u.String(), "/") + "/api/v1", hc: &http.Client{Jar: jar, Timeout: 2 * time.Minute},
		token: strings.TrimSpace(os.Getenv("NIVRA_TOKEN"))}
	if r.token == "" {
		if user == "" {
			return errors.New("set NIVRA_TOKEN (personal API token) or pass --user for password sign-in")
		}
		pw := os.Getenv("NIVRA_PASSWORD")
		if pw == "" {
			fmt.Fprintf(os.Stderr, "Password for %s: ", user)
			b, err := term.ReadPassword(int(os.Stdin.Fd()))
			fmt.Fprintln(os.Stderr)
			if err != nil {
				return fmt.Errorf("cannot read password (set NIVRA_PASSWORD or NIVRA_TOKEN): %w", err)
			}
			pw = string(b)
		}
		var login struct {
			CSRF string `json:"csrf_token"`
		}
		if err := r.do(ctx, "POST", "/auth/login", map[string]string{"username": user, "password": pw}, &login); err != nil {
			return err
		}
		r.csrf = login.CSRF
	}
	var ps struct {
		Projects []struct{ ID, Name string } `json:"projects"`
	}
	if err := r.do(ctx, "GET", "/projects", nil, &ps); err != nil {
		return err
	}
	pid := ""
	for _, p := range ps.Projects {
		if project == "" || p.Name == project || p.ID == project {
			pid = p.ID
			break
		}
	}
	if pid == "" {
		return fmt.Errorf("project %q not found", project)
	}
	body := map[string]any{"prompt": prompt, "seed": seed, "steps": steps, "cfg_scale": cfg, "fps": fps, "upscale": upscale}
	for k, v := range hd {
		body[k] = v
	}
	if model != "" {
		body["model_version"] = model
	}
	var created struct {
		JobID        string `json:"job_id"`
		GenerationID string `json:"generation_id"`
		ModelVersion string `json:"model_version"`
		Segments     int    `json:"segments"`
	}
	if err := r.do(ctx, "POST", "/projects/"+pid+"/generate", body, &created); err != nil {
		return err
	}
	if created.GenerationID != "" {
		fmt.Fprintf(os.Stderr, "long video: %d segments rendered in parallel by the connected workers (generation %s)\n",
			created.Segments, created.GenerationID)
		return r.waitGeneration(ctx, pid, created.GenerationID, out)
	}
	fmt.Fprintf(os.Stderr, "queued job %s (model %s); waiting for a worker...\n", created.JobID, created.ModelVersion)
	last := ""
	for {
		var job struct {
			Status        string `json:"status"`
			FailureReason string `json:"failure_reason"`
			Current       *struct {
				DeviceName string  `json:"device_name"`
				Progress   float64 `json:"progress"`
				Stage      string  `json:"stage"`
			} `json:"current_attempt"`
			Result *struct {
				Status     string  `json:"validation_status"`
				ArtifactID *string `json:"artifact_id"`
			} `json:"result"`
		}
		if err := r.do(ctx, "GET", "/projects/"+pid+"/jobs/"+created.JobID, nil, &job); err != nil {
			return err
		}
		line := job.Status
		if job.Current != nil {
			line = fmt.Sprintf("%s on %s %3.0f%% %s", job.Status, job.Current.DeviceName, job.Current.Progress*100, job.Current.Stage)
		}
		if line != last {
			fmt.Fprintln(os.Stderr, "  "+line)
			last = line
		}
		switch job.Status {
		case "completed":
			if job.Result == nil || job.Result.ArtifactID == nil {
				return errors.New("job completed without a video artifact")
			}
			return r.download(ctx, "/projects/"+pid+"/artifacts/"+*job.Result.ArtifactID, out)
		case "failed", "cancelled":
			return fmt.Errorf("generation %s: %s", job.Status, job.FailureReason)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
}

func (r *remote) download(ctx context.Context, path, out string) error {
	req, err := http.NewRequestWithContext(ctx, "GET", r.base+path, nil)
	if err != nil {
		return err
	}
	if r.token != "" {
		req.Header.Set("Authorization", "Bearer "+r.token)
	}
	resp, err := r.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("download: HTTP %d", resp.StatusCode)
	}
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Printf("saved %s\n", out)
	return nil
}

func copyFile(src, dst string) error {
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, b, 0o644)
}

// waitGeneration follows a long (segmented) video until it is done.
func (r *remote) waitGeneration(ctx context.Context, pid, gid, out string) error {
	last := ""
	for {
		var g struct {
			Status        string   `json:"status"`
			Stage         string   `json:"stage"`
			Progress      float64  `json:"progress"`
			SegmentsTotal int      `json:"segments_total"`
			SegmentsDone  int      `json:"segments_done"`
			Workers       []string `json:"workers"`
			ArtifactID    *string  `json:"artifact_id"`
			Failure       string   `json:"failure_reason"`
		}
		if err := r.do(ctx, "GET", "/projects/"+pid+"/generations/"+gid, nil, &g); err != nil {
			return err
		}
		line := fmt.Sprintf("%s %s %3.0f%% segments %d/%d workers %s", g.Status, g.Stage, g.Progress*100,
			g.SegmentsDone, g.SegmentsTotal, strings.Join(g.Workers, ","))
		if line != last {
			fmt.Fprintln(os.Stderr, "  "+line)
			last = line
		}
		switch g.Status {
		case "completed":
			if g.ArtifactID == nil {
				return errors.New("generation completed without a video")
			}
			return r.download(ctx, "/projects/"+pid+"/artifacts/"+*g.ArtifactID, out)
		case "failed", "cancelled":
			return fmt.Errorf("generation %s: %s", g.Status, g.Failure)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(3 * time.Second):
		}
	}
}
