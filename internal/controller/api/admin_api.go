package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/controller/projects"
	"nivra/internal/controller/scheduler"
	"nivra/internal/controller/telemetry"
	"nivra/internal/controller/workers"
	"nivra/internal/shared/protocol"
	"nivra/internal/shared/sysinfo"
	"nivra/internal/shared/versioning"
)

func telemetryInfo(projectID, deviceID, service, msg string) telemetry.LogRecord {
	p, d := projectID, deviceID
	return telemetry.LogRecord{ProjectID: &p, DeviceID: &d, Source: "controller", Service: service, Level: "info", Message: msg}
}

// ---- auth ----

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}
	if err := decodeJSON(w, r, 4096, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	ip := s.clientIP(r)
	user := strings.ToLower(req.Username)
	// Per-IP limit counts every attempt; the per-account limit counts only
	// failures, so a flood from elsewhere cannot lock out a correct login
	// for long and successful sign-ins are never throttled by it.
	if !s.loginIP.allow(rateKey(ip)) || s.loginUser.blocked(user) {
		w.Header().Set("Retry-After", "60")
		writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "too many sign-in attempts; wait a minute")
		return
	}
	token, sess, err := s.Auth.Login(r.Context(), req.Username, req.Password, ip, r.UserAgent())
	if errors.Is(err, auth.ErrInvalidCredentials) {
		s.loginUser.allow(user)
		writeErr(w, http.StatusUnauthorized, protocol.ErrUnauthorized, "invalid username or password")
		return
	}
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	s.setSessionCookie(w, token, sess.ExpiresAt)
	writeJSON(w, http.StatusOK, map[string]any{"user": sess.User, "csrf_token": sess.CSRF, "expires_at": sess.ExpiresAt})
}

func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	_ = s.Auth.Logout(r.Context(), a.Session.ID)
	http.SetCookie(w, &http.Cookie{Name: s.sessionCookieName(), Value: "", Path: "/", MaxAge: -1, HttpOnly: true, Secure: s.d.Cfg.CookieSecure})
	writeJSON(w, http.StatusOK, map[string]string{"status": "signed out"})
}

func (s *Server) handleMe(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	writeJSON(w, http.StatusOK, map[string]any{"user": a.Session.User, "csrf_token": a.Session.CSRF, "expires_at": a.Session.ExpiresAt})
}

func (s *Server) handleTokens(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	ts, err := s.Auth.ListTokens(r.Context(), a.Session.User.ID)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	if ts == nil {
		ts = []auth.APIToken{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"tokens": ts})
}

func (s *Server) handleCreateToken(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	var req struct {
		Name string `json:"name"`
		Days int    `json:"days"`
	}
	if err := decodeJSON(w, r, 1024, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	if req.Days < 0 || req.Days > 3650 {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, "days must be 0..3650 (0 = no expiry)")
		return
	}
	secret, t, err := s.Auth.CreateToken(r.Context(), a.Session.User.ID, req.Name, time.Duration(req.Days)*24*time.Hour)
	if err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"token": secret, "info": t, "note": "copy the token now; it is not shown again"})
}

func (s *Server) handleRevokeToken(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	if err := s.Auth.RevokeToken(r.Context(), a.Session.User.ID, r.PathValue("id")); err != nil {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "token not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "revoked"})
}

func (s *Server) handleUsers(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	us, err := s.Auth.ListUsers(r.Context())
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"users": us})
}

func (s *Server) handleAudit(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	rows, err := s.d.DB.Query(r.Context(), `SELECT id, ts, actor_type, actor_id, action, resource, result, remote_addr, detail
		FROM audit_events ORDER BY id DESC LIMIT 300`)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	type ev struct {
		ID       int64           `json:"id"`
		TS       time.Time       `json:"ts"`
		Actor    string          `json:"actor_type"`
		ActorID  string          `json:"actor_id"`
		Action   string          `json:"action"`
		Resource string          `json:"resource"`
		Result   string          `json:"result"`
		Remote   string          `json:"remote_addr"`
		Detail   json.RawMessage `json:"detail"`
	}
	out, err := pgx.CollectRows(rows, func(r pgx.CollectableRow) (ev, error) {
		var e ev
		err := r.Scan(&e.ID, &e.TS, &e.Actor, &e.ActorID, &e.Action, &e.Resource, &e.Result, &e.Remote, &e.Detail)
		return e, err
	})
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"events": out})
}

// ---- projects and codes ----

func (s *Server) handleProjects(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	ps, err := s.Projects.List(r.Context())
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	if ps == nil {
		ps = []projects.Project{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"projects": ps})
}

func (s *Server) handleCreateProject(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	var req struct {
		Name string `json:"name"`
	}
	if err := decodeJSON(w, r, 4096, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	p, ci, err := s.Projects.Create(r.Context(), a.Actor, req.Name)
	if err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"project": p, "code": ci})
}

func (s *Server) handleGetCode(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	ci, err := s.Projects.Code(r.Context(), a.ProjectID)
	if errors.Is(err, projects.ErrNoActiveCode) {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "enrollment is closed for this project")
		return
	}
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, ci)
}

func (s *Server) handleRotateCode(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	ci, err := s.Projects.Rotate(r.Context(), a.Actor, a.ProjectID)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, ci)
}

func (s *Server) handleCloseCode(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	if err := s.Projects.CloseEnrollment(r.Context(), a.Actor, a.ProjectID); err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "enrollment closed"})
}

// ---- overview / system ----

func (s *Server) dbHealth(ctx context.Context) map[string]any {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	start := time.Now()
	if err := s.d.DB.Ping(ctx); err != nil {
		return map[string]any{"status": "down", "error": "database unreachable"}
	}
	latency := float64(time.Since(start).Microseconds()) / 1000
	var size int64
	var version string
	_ = s.d.DB.QueryRow(ctx, `SELECT pg_database_size(current_database()), current_setting('server_version')`).Scan(&size, &version)
	st := s.d.DB.Stat()
	return map[string]any{"status": "ok", "latency_ms": latency, "size_bytes": size,
		"version": version, "pool_total": st.TotalConns(), "pool_idle": st.IdleConns(), "pool_max": st.MaxConns()}
}

func (s *Server) cloudflareStatus(ctx context.Context) map[string]any {
	u := s.d.Cfg.CloudflaredMetricsURL
	if u == "" {
		return map[string]any{"status": "not_configured"}
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(u, "/")+"/ready", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return map[string]any{"status": "unreachable"}
	}
	defer resp.Body.Close()
	var body struct {
		ReadyConnections int `json:"readyConnections"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&body)
	st := "down"
	if resp.StatusCode == http.StatusOK {
		st = "connected"
	}
	return map[string]any{"status": st, "ready_connections": body.ReadyConnections}
}

func (s *Server) handleSystem(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	writeJSON(w, http.StatusOK, s.systemInfo(r.Context()))
}

func (s *Server) systemInfo(ctx context.Context) map[string]any {
	host := sysinfo.Snapshot(s.d.Cfg.DataDir)
	n, bytes, _ := s.Artifacts.Stats(ctx)
	return map[string]any{
		"version": versioning.Version, "commit": versioning.Commit, "platform": versioning.Platform(),
		"uptime_sec": time.Since(s.Started).Seconds(), "host": host, "database": s.dbHealth(ctx),
		"cloudflare": s.cloudflareStatus(ctx), "sse_clients": s.d.Events.Subscribers(),
		"telemetry_dropped": s.Telemetry.Dropped(), "artifacts": map[string]any{"count": n, "bytes": bytes},
		"controller_fingerprint": s.d.Secrets.Fingerprint(),
	}
}

func (s *Server) handleOverview(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	ctx := r.Context()
	online, total, gpus, err := s.Workers.Counts(ctx, a.ProjectID)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	queue, err := s.Sched.QueueCounts(ctx, a.ProjectID)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	var accepted int64
	var credit float64
	_ = s.d.DB.QueryRow(ctx, `SELECT count(*), COALESCE(sum(credit),0) FROM contributions WHERE project_id=$1`, a.ProjectID).Scan(&accepted, &credit)
	p, _ := s.Projects.Get(ctx, a.ProjectID)
	out := map[string]any{"project": p, "system": s.systemInfo(ctx),
		"workers": map[string]int{"online": online, "total": total, "active_gpus": gpus},
		"queue":   queue, "contributions": map[string]any{"accepted_jobs": accepted, "credit": credit}}
	for _, e := range s.exts {
		for k, v := range e.Overview(ctx, a.ProjectID) {
			out[k] = v
		}
	}
	writeJSON(w, http.StatusOK, out)
}

// ---- workers ----

func atoiDefault(s string, d int) int {
	if n, err := strconv.Atoi(s); err == nil {
		return n
	}
	return d
}

func (s *Server) handleWorkers(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	q := r.URL.Query()
	list, total, err := s.Workers.List(r.Context(), a.ProjectID, atoiDefault(q.Get("limit"), 200), atoiDefault(q.Get("offset"), 0))
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"workers": list, "total": total})
}

func (s *Server) handleWorker(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	v, err := s.Workers.Get(r.Context(), a.ProjectID, r.PathValue("wid"))
	if errors.Is(err, workers.ErrNotFound) {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "worker not found")
		return
	}
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, v)
}

func (s *Server) handleWorkerMetrics(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	if _, err := s.Workers.Get(r.Context(), a.ProjectID, r.PathValue("wid")); err != nil {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "worker not found")
		return
	}
	mins := min(max(atoiDefault(r.URL.Query().Get("minutes"), 60), 1), 7*24*60)
	pts, err := s.Telemetry.WorkerHistory(r.Context(), r.PathValue("wid"), time.Duration(mins)*time.Minute)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"points": pts})
}

func (s *Server) handleWorkerCommands(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	cs, err := s.Workers.Commands(r.Context(), a.ProjectID, r.PathValue("wid"), 50)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"commands": cs})
}

func (s *Server) handleAllCommands(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	cs, err := s.Workers.Commands(r.Context(), a.ProjectID, "", 100)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"commands": cs})
}

func (s *Server) handleSendCommand(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	var req struct {
		Type    string          `json:"type"`
		Payload json.RawMessage `json:"payload"`
	}
	if err := decodeJSON(w, r, 8192, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	c, err := s.Workers.SendCommand(r.Context(), a.Actor, a.ProjectID, r.PathValue("wid"), req.Type, req.Payload)
	switch {
	case errors.Is(err, workers.ErrBadCommand):
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, "unknown command type")
	case errors.Is(err, workers.ErrNotFound):
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "worker not found or revoked")
	case err != nil:
		s.internalErr(w, r, err)
	default:
		writeJSON(w, http.StatusAccepted, c)
	}
}

func (s *Server) handleRename(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	var req struct {
		Name string `json:"name"`
	}
	if err := decodeJSON(w, r, 4096, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	err := s.Workers.Rename(r.Context(), a.Actor, a.ProjectID, r.PathValue("wid"), strings.TrimSpace(req.Name))
	if errors.Is(err, workers.ErrNotFound) {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "worker not found")
		return
	}
	if err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "renamed"})
}

func (s *Server) handleTrust(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	var req struct {
		Trusted bool `json:"trusted"`
	}
	if err := decodeJSON(w, r, 1024, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request")
		return
	}
	err := s.Workers.SetTrusted(r.Context(), a.Actor, a.ProjectID, r.PathValue("wid"), req.Trusted)
	if errors.Is(err, workers.ErrNotFound) {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "worker not found or revoked")
		return
	}
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"trusted": req.Trusted})
}

func (s *Server) handleRevoke(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	err := s.Workers.Revoke(r.Context(), a.Actor, a.ProjectID, r.PathValue("wid"))
	if errors.Is(err, workers.ErrNotFound) {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "worker not found or already revoked")
		return
	}
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "revoked"})
}

// ---- jobs ----

func (s *Server) handleJobs(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	q := r.URL.Query()
	f := scheduler.JobFilter{Status: q.Get("status"), JobType: q.Get("type"), RunID: q.Get("run"), Limit: atoiDefault(q.Get("limit"), 100)}
	if b := q.Get("before"); b != "" {
		if t, err := time.Parse(time.RFC3339Nano, b); err == nil {
			f.Before = &t
		}
	}
	jobs, err := s.Sched.ListJobs(r.Context(), a.ProjectID, f)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	if jobs == nil {
		jobs = []scheduler.JobView{}
	}
	counts, _ := s.Sched.QueueCounts(r.Context(), a.ProjectID)
	writeJSON(w, http.StatusOK, map[string]any{"jobs": jobs, "counts": counts, "job_types": s.Sched.Registry().Names()})
}

func (s *Server) handleCreateJob(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	var req struct {
		JobType        string          `json:"job_type"`
		Params         json.RawMessage `json:"params"`
		Priority       int             `json:"priority"`
		MaxAttempts    int             `json:"max_attempts"`
		LeaseSeconds   int             `json:"lease_seconds"`
		ReqRAMMB       int             `json:"req_ram_mb"`
		ReqVRAMMB      int             `json:"req_vram_mb"`
		ReqGPU         bool            `json:"req_gpu"`
		ReqOS          string          `json:"req_os"`
		Count          int             `json:"count"`
		IdempotencyKey string          `json:"idempotency_key"`
	}
	if err := decodeJSON(w, r, 128<<10, &req); err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrBadRequest, "malformed request: "+err.Error())
		return
	}
	if req.Count <= 0 {
		req.Count = 1
	}
	if req.Count > 1000 || (req.Count > 1 && req.IdempotencyKey != "") {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, "count must be 1..1000 and cannot be combined with idempotency_key")
		return
	}
	if req.MaxAttempts < 0 || req.MaxAttempts > 50 || req.LeaseSeconds < 0 || req.LeaseSeconds > 86400 {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, "max_attempts or lease_seconds out of range")
		return
	}
	if req.LeaseSeconds != 0 && req.LeaseSeconds < 15 {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, "lease_seconds must be >= 15")
		return
	}
	uid := a.Session.User.ID
	var ids []string
	err := core.WithTx(r.Context(), s.d.DB, func(tx pgx.Tx) error {
		for i := 0; i < req.Count; i++ {
			id, err := s.Sched.Create(r.Context(), tx, scheduler.NewJob{ProjectID: a.ProjectID, JobType: req.JobType, Params: req.Params,
				Priority: req.Priority, MaxAttempts: req.MaxAttempts, LeaseSeconds: req.LeaseSeconds, ReqRAMMB: req.ReqRAMMB,
				ReqVRAMMB: req.ReqVRAMMB, ReqGPU: req.ReqGPU, ReqOS: req.ReqOS, IdempotencyKey: req.IdempotencyKey, CreatedBy: &uid})
			if err != nil {
				return err
			}
			ids = append(ids, id)
		}
		return nil
	})
	if err != nil {
		writeErr(w, http.StatusBadRequest, protocol.ErrValidation, err.Error())
		return
	}
	s.d.Events.NotifyJobs(a.ProjectID)
	s.d.Audit(r.Context(), nil, a.Actor, "job.create", "project:"+a.ProjectID, "success", map[string]any{"type": req.JobType, "count": len(ids)})
	writeJSON(w, http.StatusCreated, map[string]any{"job_ids": ids})
}

func (s *Server) handleJob(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	v, err := s.Sched.GetJob(r.Context(), a.ProjectID, r.PathValue("jid"))
	if errors.Is(err, pgx.ErrNoRows) {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "job not found")
		return
	}
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, v)
}

func (s *Server) handleCancelJob(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	if err := s.Sched.Cancel(r.Context(), a.Actor, a.ProjectID, r.PathValue("jid")); err != nil {
		s.schedErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "cancelled"})
}

func (s *Server) handleRetryJob(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	if err := s.Sched.Retry(r.Context(), a.Actor, a.ProjectID, r.PathValue("jid")); err != nil {
		s.schedErr(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// ---- contributions ----

func (s *Server) contributionSums(r *http.Request, projectID, deviceID string) ([]protocol.ContributionSum, error) {
	rows, err := s.d.DB.Query(r.Context(), `SELECT category, count(*), COALESCE(sum(measured_seconds),0), COALESCE(sum(units),0), COALESCE(sum(credit),0)
		FROM contributions WHERE project_id=$1 AND device_id=$2 GROUP BY category ORDER BY category`, projectID, deviceID)
	if err != nil {
		return nil, err
	}
	out, err := pgx.CollectRows(rows, func(r pgx.CollectableRow) (protocol.ContributionSum, error) {
		var c protocol.ContributionSum
		err := r.Scan(&c.Category, &c.Jobs, &c.Seconds, &c.Units, &c.Credit)
		return c, err
	})
	if out == nil {
		out = []protocol.ContributionSum{}
	}
	return out, err
}

type leaderRow struct {
	DeviceID        string  `json:"device_id,omitempty"`
	Name            string  `json:"name"`
	ContributorName string  `json:"contributor_name"`
	Category        string  `json:"category"`
	Jobs            int64   `json:"jobs"`
	Seconds         float64 `json:"seconds"`
	Units           float64 `json:"units"`
	Credit          float64 `json:"credit"`
}

func (s *Server) leaderboard(ctx context.Context, projectID string, publicOnly bool) ([]leaderRow, error) {
	rows, err := s.d.DB.Query(ctx, `SELECT d.id, d.name, d.contributor_name, c.category, count(*), sum(c.measured_seconds), sum(c.units), sum(c.credit)
		FROM contributions c JOIN devices d ON d.id=c.device_id
		WHERE c.project_id=$1 AND (NOT $2 OR d.public_profile)
		GROUP BY d.id, d.name, d.contributor_name, c.category ORDER BY sum(c.credit) DESC, count(*) DESC LIMIT 500`, projectID, publicOnly)
	if err != nil {
		return nil, err
	}
	out, err := pgx.CollectRows(rows, func(r pgx.CollectableRow) (leaderRow, error) {
		var l leaderRow
		err := r.Scan(&l.DeviceID, &l.Name, &l.ContributorName, &l.Category, &l.Jobs, &l.Seconds, &l.Units, &l.Credit)
		return l, err
	})
	if out == nil {
		out = []leaderRow{}
	}
	return out, err
}

func (s *Server) handleContributions(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	rows, err := s.leaderboard(r.Context(), a.ProjectID, false)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	var recent []map[string]any
	rs, err := s.d.DB.Query(r.Context(), `SELECT c.created_at, d.name, c.category, j.job_type, c.measured_seconds, c.units, c.unit_kind, c.credit
		FROM contributions c JOIN devices d ON d.id=c.device_id JOIN jobs j ON j.id=c.job_id
		WHERE c.project_id=$1 ORDER BY c.created_at DESC LIMIT 100`, a.ProjectID)
	if err == nil {
		for rs.Next() {
			var t time.Time
			var name, cat, jt, uk string
			var secs, units, credit float64
			if rs.Scan(&t, &name, &cat, &jt, &secs, &units, &uk, &credit) == nil {
				recent = append(recent, map[string]any{"ts": t, "worker": name, "category": cat, "job_type": jt,
					"seconds": secs, "units": units, "unit_kind": uk, "credit": credit})
			}
		}
		rs.Close()
	}
	if recent == nil {
		recent = []map[string]any{}
	}
	writeJSON(w, http.StatusOK, map[string]any{"by_worker": rows, "recent": recent})
}

func (s *Server) handlePublicLeaderboard(w http.ResponseWriter, r *http.Request) {
	if !s.enrollIP.allow("lb:" + rateKey(s.clientIP(r))) {
		writeErr(w, http.StatusTooManyRequests, protocol.ErrRateLimited, "too many requests")
		return
	}
	rows, err := s.leaderboard(r.Context(), r.PathValue("pid"), true)
	if err != nil {
		writeErr(w, http.StatusNotFound, protocol.ErrNotFound, "not found")
		return
	}
	for i := range rows {
		rows[i].DeviceID = "" // do not expose identifiers publicly
	}
	writeJSON(w, http.StatusOK, map[string]any{"leaderboard": rows})
}

// ---- logs ----

func (s *Server) handleLogs(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	q := r.URL.Query()
	lq := telemetry.LogQuery{ProjectID: a.ProjectID, DeviceID: q.Get("device"), Service: q.Get("service"), RunID: q.Get("run"),
		MinLevel: q.Get("level"), Search: q.Get("q"), Limit: atoiDefault(q.Get("limit"), 200)}
	if v := q.Get("before_id"); v != "" {
		lq.BeforeID, _ = strconv.ParseInt(v, 10, 64)
	}
	for _, p := range []struct {
		key string
		dst **time.Time
	}{{"since", &lq.Since}, {"until", &lq.Until}} {
		if v := q.Get(p.key); v != "" {
			if t, err := time.Parse(time.RFC3339, v); err == nil {
				*p.dst = &t
			}
		}
	}
	logs, err := s.Telemetry.QueryLogs(r.Context(), lq)
	if err != nil {
		s.internalErr(w, r, err)
		return
	}
	if logs == nil {
		logs = []telemetry.LogRecord{}
	}
	if q.Get("format") == "ndjson" {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Content-Disposition", `attachment; filename="nivra-logs.ndjson"`)
		enc := json.NewEncoder(w)
		for _, l := range logs {
			_ = enc.Encode(l)
		}
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"logs": logs})
}

func (s *Server) handleAdminDownload(w http.ResponseWriter, r *http.Request, a *AdminCtx) {
	s.serveArtifact(w, r, a.ProjectID, r.PathValue("aid"))
}
