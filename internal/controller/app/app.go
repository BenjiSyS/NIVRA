// Package app implements the controller startup sequence and lifecycle.
package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"nivra/database"
	"nivra/internal/controller/api"
	"nivra/internal/controller/artifacts"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/controller/enrollment"
	"nivra/internal/controller/events"
	"nivra/internal/controller/projects"
	"nivra/internal/controller/scheduler"
	"nivra/internal/controller/secrets"
	"nivra/internal/controller/telemetry"
	"nivra/internal/controller/workers"
	"nivra/internal/shared/config"
	"nivra/internal/shared/lockfile"
	"nivra/internal/shared/versioning"
	"nivra/internal/webui"
)

// DataSubdirs are created under the controller data directory.
var DataSubdirs = []string{"artifacts", "logs", "secrets", "backups", "tmp", "checkpoints", "datasets", "cache"}

// Runtime holds the constructed controller services.
type Runtime struct {
	Deps      *core.Deps
	Auth      *auth.Service
	Projects  *projects.Service
	Enroll    *enrollment.Service
	Registry  *scheduler.Registry
	Sched     *scheduler.Scheduler
	Workers   *workers.Service
	Telemetry *telemetry.Service
	Artifacts *artifacts.Store
	Server    *api.Server
	lock      *lockfile.Lock
	logFile   io.Closer
}

// ExtensionFactory builds optional subsystems (e.g. distributed training)
// once core services exist.
type ExtensionFactory func(rt *Runtime) (api.Extension, func(context.Context), error)

// Options customise startup.
type Options struct {
	Extensions []ExtensionFactory
	// Ready is called with the bound address once the API is listening.
	Ready func(addr net.Addr)
	// LogWriter overrides stderr for console logs.
	LogWriter io.Writer
}

// Connect opens a database pool, retrying while PostgreSQL starts.
func Connect(ctx context.Context, cfg config.Controller, log *slog.Logger) (*pgxpool.Pool, error) {
	if cfg.DatabaseURL == "" {
		return nil, errors.New("no database configured: set NIVRA_DATABASE_URL or <data_dir>/secrets/database.env")
	}
	pc, err := pgxpool.ParseConfig(cfg.DatabaseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid database URL: %w", err)
	}
	pc.MaxConns = int32(cfg.DBMaxConns)
	pc.MinConns = 1
	pc.MaxConnIdleTime = 5 * time.Minute
	pc.ConnConfig.RuntimeParams["application_name"] = "nivra-controller"
	var lastErr error
	for attempt := 1; attempt <= 15; attempt++ {
		pool, err := pgxpool.NewWithConfig(ctx, pc)
		if err == nil {
			pctx, cancel := context.WithTimeout(ctx, 5*time.Second)
			err = pool.Ping(pctx)
			cancel()
			if err == nil {
				return pool, nil
			}
			pool.Close()
		}
		lastErr = err
		if log != nil {
			log.Warn("database not reachable yet; retrying", "attempt", attempt, "err", err)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	return nil, fmt.Errorf("database unreachable: %w", lastErr)
}

// EnsureDirs creates the runtime directory layout with restrictive permissions.
func EnsureDirs(dataDir string) error {
	for _, d := range DataSubdirs {
		perm := os.FileMode(0o750)
		if d == "secrets" {
			perm = 0o700
		}
		if err := os.MkdirAll(filepath.Join(dataDir, d), perm); err != nil {
			return err
		}
	}
	// Verify the directory is writable.
	probe := filepath.Join(dataDir, "tmp", ".write-probe")
	if err := os.WriteFile(probe, []byte("ok"), 0o600); err != nil {
		return fmt.Errorf("data directory %s is not writable: %w", dataDir, err)
	}
	return os.Remove(probe)
}

// NewLogger writes JSON logs to the console and <data_dir>/logs/controller.log.
func NewLogger(dataDir string, console io.Writer) (*slog.Logger, io.Closer, error) {
	path := filepath.Join(dataDir, "logs", "controller.log")
	if st, err := os.Stat(path); err == nil && st.Size() > 50<<20 {
		_ = os.Rename(path, path+".1")
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o640)
	if err != nil {
		return nil, nil, err
	}
	if console == nil {
		console = os.Stderr
	}
	h := slog.NewTextHandler(io.MultiWriter(console, f), &slog.HandlerOptions{Level: slog.LevelInfo})
	return slog.New(h), f, nil
}

// Build performs startup steps 1–8 (validation, storage, database,
// migrations, secrets, project-code verification) and constructs services.
func Build(ctx context.Context, cfg config.Controller, opts Options) (*Runtime, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("configuration: %w", err)
	}
	if err := EnsureDirs(cfg.DataDir); err != nil {
		return nil, err
	}
	lk, err := lockfile.Acquire(filepath.Join(cfg.DataDir, "controller.lock"))
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			lk.Release()
		}
	}()
	baseLog, logFile, err := NewLogger(cfg.DataDir, opts.LogWriter)
	if err != nil {
		return nil, err
	}
	defer func() {
		if !ok {
			logFile.Close()
		}
	}()
	baseLog.Info("starting Nivra controller", "version", versioning.String(), "data_dir", cfg.DataDir)
	sec, created, err := secrets.LoadOrCreate(cfg.DataDir)
	if err != nil {
		return nil, fmt.Errorf("secrets: %w", err)
	}
	if created {
		baseLog.Warn("generated new controller secrets; back up this file securely", "path", secrets.Path(cfg.DataDir))
	}
	pool, err := Connect(ctx, cfg, baseLog)
	if err != nil {
		return nil, err
	}
	st, err := database.Inspect(ctx, pool)
	if err != nil {
		pool.Close()
		return nil, err
	}
	if len(st.Pending) > 0 {
		if len(st.Applied) > 0 {
			baseLog.Warn("applying schema migrations to an existing database; ensure a backup exists (see docs/installation/UPGRADES.md)",
				"pending", len(st.Pending))
		}
		applied, err := database.Apply(ctx, pool)
		if err != nil {
			pool.Close()
			return nil, fmt.Errorf("migrations: %w", err)
		}
		baseLog.Info("migrations applied", "versions", applied)
	}

	tel := telemetry.New(pool, cfg.WorkerMetricsRetainDays, cfg.LogRetainDays)
	log := slog.New(telemetry.NewHandler(baseLog.Handler(), tel, "controller"))
	d := &core.Deps{DB: pool, Cfg: cfg, Secrets: sec, Events: events.NewHub(), Log: log}
	reg := scheduler.NewRegistry()
	scheduler.RegisterBuiltins(reg)
	store, err := artifacts.New(d)
	if err != nil {
		pool.Close()
		return nil, err
	}
	rt := &Runtime{Deps: d, Auth: auth.New(d), Projects: projects.New(d), Enroll: enrollment.New(d), Registry: reg,
		Telemetry: tel, Artifacts: store, lock: lk, logFile: logFile}
	rt.Sched = scheduler.New(d, reg)
	rt.Workers = workers.New(d, rt.Sched, tel)
	tel.OnLog(func(r telemetry.LogRecord) {
		if r.Level == "warn" || r.Level == "error" {
			pid := ""
			if r.ProjectID != nil {
				pid = *r.ProjectID
			}
			d.Events.Publish(events.Event{Type: "log", ProjectID: pid, Data: r})
		}
	})
	n, err := rt.Projects.VerifyCodes(ctx)
	if err != nil {
		pool.Close()
		return nil, fmt.Errorf("restore project codes: %w", err)
	}
	log.Info("projects loaded", "active_codes", n)
	rt.Server = api.New(api.Config{Deps: d, Auth: rt.Auth, Projects: rt.Projects, Enroll: rt.Enroll, Sched: rt.Sched,
		Workers: rt.Workers, Telemetry: tel, Artifacts: store, UI: webui.FS()})
	ok = true
	return rt, nil
}

// Run builds the controller and serves until ctx is cancelled, then shuts
// down every background loop, flushes telemetry and closes the database.
func Run(ctx context.Context, cfg config.Controller, opts Options) error {
	rt, err := Build(ctx, cfg, opts)
	if err != nil {
		return err
	}
	return rt.Serve(ctx, opts)
}

// Serve starts background loops and the HTTP server; it releases the
// database and instance lock when ctx is cancelled.
func (rt *Runtime) Serve(ctx context.Context, opts Options) error {
	cfg := rt.Deps.Cfg
	defer rt.lock.Release()
	defer rt.logFile.Close()
	defer rt.Deps.DB.Close()
	log := rt.Deps.Log
	var err error

	// Restore scheduler state: expire leases that lapsed while we were down.
	if n, err := rt.Sched.ReapOnce(ctx); err == nil && n > 0 {
		log.Info("expired leases from before restart", "count", n)
	}

	bg, cancelBG := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	start := func(f func(context.Context)) {
		wg.Add(1)
		go func() { defer wg.Done(); f(bg) }()
	}
	for _, f := range opts.Extensions {
		ext, loop, err := f(rt)
		if err != nil {
			cancelBG()
			return err
		}
		if ext != nil {
			rt.Server.AddExtension(ext)
		}
		if loop != nil {
			start(loop)
		}
	}
	start(func(c context.Context) { rt.Telemetry.Run(c, log) })
	start(func(c context.Context) { rt.Sched.RunReaper(c, 5*time.Second) })
	start(rt.Sched.RunValidationLoop)
	start(rt.Workers.RunPresence)
	start(func(c context.Context) {
		t := time.NewTicker(time.Hour)
		defer t.Stop()
		for {
			_ = rt.Enroll.PurgeExpired(c)
			_ = rt.Auth.PurgeExpired(c)
			_ = rt.Artifacts.PurgeStaleUploads(c)
			select {
			case <-c.Done():
				return
			case <-t.C:
			}
		}
	})

	err = rt.Server.ListenAndServe(ctx, cfg.ListenAddr, func(a net.Addr) {
		log.Info("Nivra Studio ready", "url", "http://"+a.String()+"/", "api", "http://"+a.String()+"/api/v1/health",
			"controller_fingerprint", rt.Deps.Secrets.Fingerprint())
		if opts.Ready != nil {
			opts.Ready(a)
		}
	})
	log.Info("shutting down controller")
	cancelBG()
	wg.Wait()
	log.Info("controller stopped cleanly")
	return err
}

// Close releases the database, log file and instance lock of a Runtime that
// was built but not served (CLI tools).
func (rt *Runtime) Close() {
	rt.Deps.DB.Close()
	rt.logFile.Close()
	rt.lock.Release()
}
