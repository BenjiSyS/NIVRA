// Package doctor implements `nivra doctor` (diagnostics) and `nivra repair`
// (conservative, data-preserving fixes).
package doctor

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"nivra/database"
	"nivra/internal/controller/app"
	"nivra/internal/controller/secrets"
	"nivra/internal/shared/config"
	"nivra/internal/shared/sysinfo"
	"nivra/internal/shared/versioning"
	"nivra/internal/worker/hardware"
	"nivra/internal/worker/identity"
)

// Options select what to check.
type Options struct {
	Role       string // controller | worker | "" (both, where configured)
	ConfigPath string
	Repair     bool
}

type reporter struct {
	w      io.Writer
	failed int
}

func (r *reporter) ok(name, msg string)   { fmt.Fprintf(r.w, "  [ OK ] %-28s %s\n", name, msg) }
func (r *reporter) warn(name, msg string) { fmt.Fprintf(r.w, "  [WARN] %-28s %s\n", name, msg) }
func (r *reporter) fail(name, msg string) {
	r.failed++
	fmt.Fprintf(r.w, "  [FAIL] %-28s %s\n", name, msg)
}
func (r *reporter) fixed(name, msg string) { fmt.Fprintf(r.w, "  [FIX ] %-28s %s\n", name, msg) }

// Run executes the checks and returns true if none failed.
func Run(ctx context.Context, w io.Writer, o Options) bool {
	r := &reporter{w: w}
	fmt.Fprintf(w, "Nivra diagnostics — %s\n\nSystem\n", versioning.String())
	h := sysinfo.Snapshot(".")
	r.ok("platform", fmt.Sprintf("%s/%s, %s, %d threads", runtime.GOOS, runtime.GOARCH, h.CPUModel, h.Threads))
	if h.MemTotal > 0 {
		r.ok("memory", fmt.Sprintf("%d MiB total, %d MiB available", h.MemTotal>>20, h.MemAvail>>20))
	}
	for _, t := range []string{"git"} {
		if p, err := exec.LookPath(t); err == nil {
			r.ok("tool "+t, p)
		} else {
			r.warn("tool "+t, "not found on PATH (needed for some features)")
		}
	}
	ctrlPath, workPath := o.ConfigPath, o.ConfigPath
	if ctrlPath == "" {
		ctrlPath = filepath.Join("config", "controller.json")
	}
	if workPath == "" {
		workPath = filepath.Join("config", "worker.json")
	}
	workerRole := o.Role == "worker" || (o.Role == "" && exists(workPath))
	if !workerRole {
		checkFFmpeg(ctx, r, "", false)
	}
	if o.Role == "controller" || (o.Role == "" && exists(ctrlPath)) {
		fmt.Fprintln(w, "\nController")
		checkController(ctx, r, ctrlPath, o.Repair)
	}
	if workerRole {
		fmt.Fprintln(w, "\nWorker")
		checkWorker(ctx, r, workPath, o.Repair)
	}
	if o.Role == "" && !exists(ctrlPath) && !exists(workPath) {
		r.warn("configuration", "no config/controller.json or config/worker.json found; run the installer")
	}
	fmt.Fprintf(w, "\n%d check(s) failed.\n", r.failed)
	return r.failed == 0
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

func checkController(ctx context.Context, r *reporter, path string, repair bool) {
	cfg, err := config.LoadController(path)
	if err != nil {
		r.fail("config", err.Error())
		return
	}
	r.ok("config", path)
	missing := []string{}
	for _, d := range app.DataSubdirs {
		if !exists(filepath.Join(cfg.DataDir, d)) {
			missing = append(missing, d)
		}
	}
	if len(missing) > 0 {
		if repair {
			if err := app.EnsureDirs(cfg.DataDir); err != nil {
				r.fail("data directories", err.Error())
			} else {
				r.fixed("data directories", "created "+strings.Join(missing, ", "))
			}
		} else {
			r.fail("data directories", "missing under "+cfg.DataDir+": "+strings.Join(missing, ", ")+" (run nivra repair)")
		}
	} else {
		r.ok("data directories", cfg.DataDir)
	}
	if free, err := sysinfo.DiskFree(cfg.DataDir); err == nil {
		if free < 5<<30 {
			r.warn("disk space", fmt.Sprintf("only %d MiB free", free>>20))
		} else {
			r.ok("disk space", fmt.Sprintf("%d GiB free", free>>30))
		}
	}
	sp := secrets.Path(cfg.DataDir)
	if _, err := secrets.Load(sp); err != nil {
		if os.IsNotExist(err) {
			r.warn("secrets", "not created yet (created on first start); back it up once it exists")
		} else {
			r.fail("secrets", err.Error()+" — do NOT delete it; restore from backup")
		}
	} else {
		r.ok("secrets", sp)
		if runtime.GOOS != "windows" {
			if st, err := os.Stat(sp); err == nil && st.Mode().Perm()&0o077 != 0 {
				if repair {
					_ = os.Chmod(sp, 0o600)
					r.fixed("secrets permissions", "set to 0600")
				} else {
					r.fail("secrets permissions", fmt.Sprintf("%v is readable by others (run nivra repair)", st.Mode().Perm()))
				}
			}
		}
	}
	if cfg.DatabaseURL == "" {
		r.fail("database", "no database configured (secrets/database.env or NIVRA_DATABASE_URL)")
	} else {
		cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		pool, err := app.Connect(cctx, cfg, nil)
		cancel()
		if err != nil {
			r.fail("database", "unreachable: "+err.Error())
		} else {
			st, err := database.Inspect(ctx, pool)
			switch {
			case err != nil:
				r.fail("migrations", err.Error())
			case len(st.Pending) > 0 && repair:
				if v, err := database.Apply(ctx, pool); err != nil {
					r.fail("migrations", err.Error())
				} else {
					r.fixed("migrations", fmt.Sprintf("applied %v", v))
				}
			case len(st.Pending) > 0:
				r.warn("migrations", fmt.Sprintf("%d pending (applied automatically at controller start)", len(st.Pending)))
			default:
				r.ok("database", fmt.Sprintf("connected; schema at version %v", st.Applied[len(st.Applied)-1]))
			}
			pool.Close()
		}
	}
	ln, err := net.Listen("tcp", cfg.ListenAddr)
	if err != nil {
		r.warn("listen address", cfg.ListenAddr+" is in use (controller already running?)")
	} else {
		ln.Close()
		r.ok("listen address", cfg.ListenAddr+" available")
	}
	checkPython(ctx, r, cfg.PythonPath, cfg.AIPackageDir, "controller AI tasks")
}

func checkPython(ctx context.Context, r *reporter, python, aiDir, purpose string) {
	p := hardware.ProbeAI(ctx, python, aiDir, "")
	if p.OK {
		r.ok("python/torch", fmt.Sprintf("python %s, torch %s, cuda=%v (%s)", p.Python, p.Torch, p.CUDA, purpose))
		if p.CUDA {
			r.ok("gpu (torch)", fmt.Sprintf("%s, %d MiB, compute %s", p.CUDADevice, p.CUDAVRAMMB, p.ComputeCap))
			if cc := p.ComputeCap; cc != "" && cc < "7.0" {
				r.warn("gpu capability", "compute capability below 7.0: FP16 tensor cores unavailable; expect slow training")
			}
		}
	} else {
		r.warn("python/torch", "unavailable for "+purpose+": "+p.Error)
	}
}

func checkWorker(ctx context.Context, r *reporter, path string, repair bool) {
	cfg, err := config.LoadWorker(path)
	if err != nil {
		r.fail("config", err.Error())
		return
	}
	r.ok("config", path)
	for _, d := range []string{"", "work", "cache", "outbox", "logs"} {
		p := filepath.Join(cfg.DataDir, d)
		if !exists(p) {
			if repair {
				_ = os.MkdirAll(p, 0o700)
				r.fixed("data directory", p)
			} else {
				r.warn("data directory", p+" missing (created on start)")
			}
		}
	}
	id, err := identity.NewStore(cfg.DataDir).Load()
	switch {
	case err != nil:
		r.fail("enrollment", "identity file unreadable: "+err.Error())
	case id == nil:
		r.warn("enrollment", "not enrolled; open the worker website and enter a project code")
	default:
		r.ok("enrollment", fmt.Sprintf("project %q at %s as %q", id.ProjectName, id.ControllerURL, id.DeviceName))
		if _, err := config.CheckControllerURL(id.ControllerURL, cfg.AllowInsecureHTTP); err != nil {
			r.fail("controller URL", err.Error())
		} else {
			cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()
			if err := healthCheck(cctx, strings.TrimRight(id.ControllerURL, "/")+"/api/v1/health"); err != nil {
				r.fail("controller reachability", err.Error())
			} else {
				r.ok("controller reachability", "health endpoint answered")
			}
		}
	}
	gs := hardware.GPUs(ctx)
	if len(gs) == 0 {
		r.ok("gpu", "no NVIDIA GPU detected (CPU-only worker)")
	} else {
		for _, g := range gs {
			r.ok(fmt.Sprintf("gpu %d", g.Index), fmt.Sprintf("%s, %d MiB, driver %s", g.Name, g.VRAMMB, g.Driver))
		}
	}
	checkFFmpeg(ctx, r, cfg.FFmpegPath, true)
	checkPython(ctx, r, cfg.PythonPath, cfg.AIPackageDir, "worker AI jobs")
	ln, err := net.Listen("tcp", cfg.UIListenAddr)
	if err != nil {
		r.warn("worker website port", cfg.UIListenAddr+" in use (worker already running?)")
	} else {
		ln.Close()
		r.ok("worker website port", cfg.UIListenAddr+" available")
	}
}

// checkFFmpeg reports whether videos can be saved here. Without ffmpeg a worker
// cannot run video jobs, so that is a failure for the worker role and a warning
// elsewhere.
func checkFFmpeg(ctx context.Context, r *reporter, configured string, mustHave bool) {
	p := hardware.ProbeFFmpeg(ctx, configured)
	if p.OK {
		r.ok("tool ffmpeg", p.FFmpegPath+" (version "+p.Version+"); ffprobe "+p.FFprobePath)
		return
	}
	msg := p.Error + " (install it with winget install Gyan.FFmpeg, or set ffmpeg_path in the worker config)"
	if mustHave {
		r.fail("tool ffmpeg", msg)
	} else {
		r.warn("tool ffmpeg", msg)
	}
}
