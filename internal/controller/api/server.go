// Package api serves the controller's HTTP API, real-time events and the
// embedded Nivra Studio website.
package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"

	"nivra/internal/controller/artifacts"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/controller/enrollment"
	"nivra/internal/controller/projects"
	"nivra/internal/controller/scheduler"
	"nivra/internal/controller/telemetry"
	"nivra/internal/controller/workers"
	"nivra/internal/shared/protocol"
)

// Extension lets other subsystems (training, generation) add routes and
// overview fields without the api package importing them.
type Extension interface {
	Register(r *Router)
	Overview(ctx context.Context, projectID string) map[string]any
}

// Server bundles every controller service behind HTTP.
type Server struct {
	d         *core.Deps
	Auth      *auth.Service
	Projects  *projects.Service
	Enroll    *enrollment.Service
	Sched     *scheduler.Scheduler
	Workers   *workers.Service
	Telemetry *telemetry.Service
	Artifacts *artifacts.Store
	UI        fs.FS
	Started   time.Time

	exts []Extension

	enrollIP      *limiter // enrollment/begin: guards project-code guessing
	enrollGlobal  *limiter
	completeIP    *limiter // enrollment/complete: needs a valid challenge anyway
	sessionDevice *limiter // worker/session per device
	sessionIP     *limiter // worker/session per source address (NAT-friendly)
	loginIP       *limiter
	loginUser     *limiter
	workerRate    *limiter
	adminRate     *limiter

	httpSrv *http.Server
}

// Config wires services into a Server.
type Config struct {
	Deps      *core.Deps
	Auth      *auth.Service
	Projects  *projects.Service
	Enroll    *enrollment.Service
	Sched     *scheduler.Scheduler
	Workers   *workers.Service
	Telemetry *telemetry.Service
	Artifacts *artifacts.Store
	UI        fs.FS
}

// New creates a server.
func New(c Config) *Server {
	cfg := c.Deps.Cfg
	return &Server{
		d: c.Deps, Auth: c.Auth, Projects: c.Projects, Enroll: c.Enroll, Sched: c.Sched, Workers: c.Workers,
		Telemetry: c.Telemetry, Artifacts: c.Artifacts, UI: c.UI, Started: time.Now(),
		enrollIP:      perMinute(cfg.EnrollAttemptsPerMinute),
		enrollGlobal:  perMinute(cfg.EnrollAttemptsPerMinute * 20),
		completeIP:    perMinute(cfg.EnrollAttemptsPerMinute * 6),
		sessionDevice: perMinute(12),
		sessionIP:     perMinute(1200),
		loginIP:       perMinute(cfg.LoginAttemptsPerMinute),
		loginUser:     perMinute(cfg.LoginAttemptsPerMinute),
		workerRate:    newLimiter(float64(cfg.WorkerRequestsPerSecond), cfg.WorkerRequestsPerSecond*2),
		adminRate:     newLimiter(50, 100),
	}
}

// AddExtension registers an extension before Handler is called.
func (s *Server) AddExtension(e Extension) { s.exts = append(s.exts, e) }

// Router is a thin wrapper used by extensions to register routes with the
// standard middleware chains.
type Router struct {
	s   *Server
	mux *http.ServeMux
}

// Admin registers a project-scoped administrator route under
// /api/v1/projects/{pid}/... requiring at least role.
func (r *Router) Admin(method, pattern, role string, h func(w http.ResponseWriter, req *http.Request, a *AdminCtx)) {
	r.mux.Handle(method+" /api/v1/projects/{pid}"+pattern, r.s.admin(role, true, h))
}

// Handler builds the full HTTP handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	s.routes(mux)
	for _, e := range s.exts {
		e.Register(&Router{s: s, mux: mux})
	}
	return s.recoverer(s.securityHeaders(mux))
}

// ListenAndServe serves until ctx is cancelled, then shuts down gracefully.
func (s *Server) ListenAndServe(ctx context.Context, addr string, ready func(net.Addr)) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w (is another controller already running?)", addr, err)
	}
	s.httpSrv = &http.Server{
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       5 * time.Minute, // artifact chunks
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    32 << 10,
		ErrorLog:          slog.NewLogLogger(s.d.Log.Handler(), slog.LevelWarn),
	}
	if ready != nil {
		ready(ln.Addr())
	}
	errCh := make(chan error, 1)
	go func() { errCh <- s.httpSrv.Serve(ln) }()
	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
		sctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		return s.httpSrv.Shutdown(sctx)
	}
}

// ---- helpers ----

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, protocol.ErrorBody{Error: protocol.ErrorDetail{Code: code, Message: msg}})
}

// internalErr logs the detail and returns a generic message.
func (s *Server) internalErr(w http.ResponseWriter, r *http.Request, err error) {
	if errors.Is(err, context.Canceled) {
		return
	}
	s.d.Log.Error("request failed", "method", r.Method, "path", r.URL.Path, "err", err)
	writeErr(w, http.StatusInternalServerError, protocol.ErrInternal, "internal error")
}

// decodeJSON strictly decodes a bounded JSON body.
func decodeJSON(w http.ResponseWriter, r *http.Request, max int64, v any) error {
	r.Body = http.MaxBytesReader(w, r.Body, max)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return err
	}
	if _, err := io.Copy(io.Discard, io.LimitReader(r.Body, 1)); err != nil {
		return err
	}
	return nil
}

func (s *Server) clientIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	if s.d.Cfg.TrustProxyHeaders {
		if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
			if cf := strings.TrimSpace(r.Header.Get("CF-Connecting-IP")); net.ParseIP(cf) != nil {
				return cf
			}
			if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
				first, _, _ := strings.Cut(xff, ",")
				if ip := strings.TrimSpace(first); net.ParseIP(ip) != nil {
					return ip
				}
			}
		}
	}
	return host
}

func (s *Server) securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "same-origin")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
		if s.d.Cfg.CookieSecure {
			h.Set("Strict-Transport-Security", "max-age=31536000")
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				if v == http.ErrAbortHandler {
					panic(v)
				}
				s.d.Log.Error("panic in handler", "path", r.URL.Path, "panic", fmt.Sprint(v))
				writeErr(w, http.StatusInternalServerError, protocol.ErrInternal, "internal error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func (s *Server) routes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/v1/health", s.handleHealth)

	// Worker enrollment and sessions (unauthenticated, rate limited).
	mux.HandleFunc("POST /api/v1/enrollment/begin", s.handleEnrollBegin)
	mux.HandleFunc("POST /api/v1/enrollment/complete", s.handleEnrollComplete)
	mux.HandleFunc("POST /api/v1/worker/session", s.handleWorkerSession)

	// Worker routes (bearer session).
	mux.Handle("POST /api/v1/worker/heartbeat", s.worker(s.handleHeartbeat))
	mux.Handle("GET /api/v1/worker/me", s.worker(s.handleWorkerMe))
	mux.Handle("GET /api/v1/worker/jobs/next", s.worker(s.handleJobNext))
	mux.Handle("POST /api/v1/worker/jobs/{attempt}/ack", s.worker(s.handleJobAck))
	mux.Handle("POST /api/v1/worker/jobs/{attempt}/progress", s.worker(s.handleJobProgress))
	mux.Handle("POST /api/v1/worker/jobs/{attempt}/complete", s.worker(s.handleJobComplete))
	mux.Handle("POST /api/v1/worker/jobs/{attempt}/fail", s.worker(s.handleJobFail))
	mux.Handle("GET /api/v1/worker/commands", s.worker(s.handleCommands))
	mux.Handle("POST /api/v1/worker/commands/{id}/ack", s.worker(s.handleCommandAck))
	mux.Handle("POST /api/v1/worker/logs", s.worker(s.handleWorkerLogs))
	mux.Handle("POST /api/v1/artifacts/uploads", s.worker(s.handleUploadBegin))
	mux.Handle("GET /api/v1/artifacts/uploads/{id}", s.worker(s.handleUploadStatus))
	mux.Handle("PUT /api/v1/artifacts/uploads/{id}", s.worker(s.handleUploadChunk))
	mux.Handle("POST /api/v1/artifacts/uploads/{id}/complete", s.worker(s.handleUploadFinish))
	mux.Handle("GET /api/v1/artifacts/{id}", s.worker(s.handleWorkerDownload))

	// Administrator routes (cookie session + CSRF).
	mux.HandleFunc("POST /api/v1/auth/login", s.handleLogin)
	mux.Handle("POST /api/v1/auth/logout", s.admin(auth.RoleViewer, false, s.handleLogout))
	mux.Handle("GET /api/v1/auth/me", s.admin(auth.RoleViewer, false, s.handleMe))
	mux.Handle("GET /api/v1/projects", s.admin(auth.RoleViewer, false, s.handleProjects))
	mux.Handle("POST /api/v1/projects", s.admin(auth.RoleAdmin, false, s.handleCreateProject))
	mux.Handle("GET /api/v1/system", s.admin(auth.RoleViewer, false, s.handleSystem))
	mux.Handle("GET /api/v1/users", s.admin(auth.RoleAdmin, false, s.handleUsers))
	mux.Handle("GET /api/v1/audit", s.admin(auth.RoleAdmin, false, s.handleAudit))

	p := func(method, pattern, role string, h func(http.ResponseWriter, *http.Request, *AdminCtx)) {
		mux.Handle(method+" /api/v1/projects/{pid}"+pattern, s.admin(role, true, h))
	}
	p("GET", "/code", auth.RoleOperator, s.handleGetCode)
	p("POST", "/code/rotate", auth.RoleAdmin, s.handleRotateCode)
	p("POST", "/code/close", auth.RoleAdmin, s.handleCloseCode)
	p("GET", "/overview", auth.RoleViewer, s.handleOverview)
	p("GET", "/events", auth.RoleViewer, s.handleEvents)
	p("GET", "/workers", auth.RoleViewer, s.handleWorkers)
	p("GET", "/workers/{wid}", auth.RoleViewer, s.handleWorker)
	p("GET", "/workers/{wid}/metrics", auth.RoleViewer, s.handleWorkerMetrics)
	p("GET", "/workers/{wid}/commands", auth.RoleViewer, s.handleWorkerCommands)
	p("POST", "/workers/{wid}/commands", auth.RoleOperator, s.handleSendCommand)
	p("POST", "/workers/{wid}/rename", auth.RoleOperator, s.handleRename)
	p("POST", "/workers/{wid}/revoke", auth.RoleAdmin, s.handleRevoke)
	p("GET", "/commands", auth.RoleViewer, s.handleAllCommands)
	p("GET", "/jobs", auth.RoleViewer, s.handleJobs)
	p("POST", "/jobs", auth.RoleOperator, s.handleCreateJob)
	p("GET", "/jobs/{jid}", auth.RoleViewer, s.handleJob)
	p("POST", "/jobs/{jid}/cancel", auth.RoleOperator, s.handleCancelJob)
	p("POST", "/jobs/{jid}/retry", auth.RoleOperator, s.handleRetryJob)
	p("GET", "/contributions", auth.RoleViewer, s.handleContributions)
	p("GET", "/logs", auth.RoleViewer, s.handleLogs)
	p("GET", "/artifacts/{aid}", auth.RoleViewer, s.handleAdminDownload)

	// Public leaderboard (only devices that opted in).
	mux.HandleFunc("GET /api/v1/public/projects/{pid}/leaderboard", s.handlePublicLeaderboard)

	// Unknown API paths return JSON 404 instead of the SPA.
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "no such API route")
	})
	if s.UI != nil {
		mux.Handle("/", spaHandler(s.UI, "studio.html"))
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	status, code := "ok", http.StatusOK
	if err := s.d.DB.Ping(ctx); err != nil {
		status, code = "degraded", http.StatusServiceUnavailable
	}
	writeJSON(w, code, map[string]any{"status": status, "service": "nivra-controller",
		"protocol_version": protocol.Version, "time": time.Now().UTC()})
}

// WriteJSON writes a JSON response (for extensions).
func WriteJSON(w http.ResponseWriter, status int, v any) { writeJSON(w, status, v) }

// WriteErr writes a JSON error (for extensions).
func WriteErr(w http.ResponseWriter, status int, code, msg string) { writeErr(w, status, code, msg) }

// DecodeJSON strictly decodes a bounded body (for extensions).
func DecodeJSON(w http.ResponseWriter, r *http.Request, max int64, v any) error {
	return decodeJSON(w, r, max, v)
}

// Internal logs err and writes a generic 500 (for extensions).
func (r *Router) Internal(w http.ResponseWriter, req *http.Request, err error) {
	r.s.internalErr(w, req, err)
}
