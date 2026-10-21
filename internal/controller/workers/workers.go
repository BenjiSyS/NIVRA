// Package workers tracks enrolled devices: heartbeats, capabilities,
// presence, administrative state and reliable commands.
package workers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/controller/events"
	"nivra/internal/controller/scheduler"
	"nivra/internal/controller/telemetry"
	"nivra/internal/shared/protocol"
)

// Dashboard-visible worker states.
const (
	StateOnline       = "online"
	StateBusy         = "busy"
	StateIdle         = "idle"
	StateDegraded     = "degraded"
	StateDisconnected = "disconnected"
	StateSuspended    = "suspended"
	StateRevoked      = "revoked"
)

// Errors.
var (
	ErrNotFound      = errors.New("worker not found")
	ErrBadCommand    = errors.New("invalid command")
	ErrBadTransition = errors.New("invalid command state transition")
)

// Service manages workers.
type Service struct {
	d     *core.Deps
	sched *scheduler.Scheduler
	tel   *telemetry.Service

	mu       sync.Mutex
	lastSeen map[string]string // device -> last published state
}

// New creates the workers service.
func New(d *core.Deps, s *scheduler.Scheduler, t *telemetry.Service) *Service {
	return &Service{d: d, sched: s, tel: t, lastSeen: map[string]string{}}
}

func (s *Service) disconnectAfter() time.Duration {
	return time.Duration(s.d.Cfg.HeartbeatIntervalSec*s.d.Cfg.DisconnectAfterMissed) * time.Second
}

// ComputeState derives the dashboard state from stored facts.
func ComputeState(status string, paused bool, lastSeen *time.Time, reported string, busy bool, disconnectAfter time.Duration) string {
	switch {
	case status == "revoked":
		return StateRevoked
	case lastSeen == nil || time.Since(*lastSeen) > disconnectAfter:
		return StateDisconnected
	case paused:
		return StateSuspended
	case reported == protocol.WorkerDegraded:
		return StateDegraded
	case busy:
		return StateBusy
	default:
		return StateIdle
	}
}

// Heartbeat processes a worker heartbeat.
func (s *Service) Heartbeat(ctx context.Context, sessionID, projectID, deviceID string, req protocol.HeartbeatRequest) (protocol.HeartbeatResponse, error) {
	resp := protocol.HeartbeatResponse{ServerTime: time.Now().UTC(), NextHeartbeatSec: s.d.Cfg.HeartbeatIntervalSec, InvalidAttempts: []string{}}
	switch req.State {
	case protocol.WorkerIdle, protocol.WorkerBusy, protocol.WorkerPaused, protocol.WorkerDegraded, protocol.WorkerStopping:
	default:
		return resp, fmt.Errorf("unknown worker state %q", req.State)
	}
	if len(req.Attempts) > 16 {
		return resp, errors.New("too many attempts reported")
	}
	err := s.d.DB.QueryRow(ctx, `UPDATE devices SET last_seen_at=now(), reported_state=$2,
			agent_version=CASE WHEN $3 <> '' THEN $3 ELSE agent_version END
		WHERE id=$1 RETURNING paused`, deviceID, req.State, truncate(req.AgentVersion, 64)).Scan(&resp.Paused)
	if err != nil {
		return resp, err
	}
	if err := s.d.DB.QueryRow(ctx, `UPDATE worker_sessions SET last_seen_at=now(), connection_state='connected' WHERE id=$1
		RETURNING expires_at`, sessionID).Scan(&resp.SessionExpires); err != nil {
		return resp, err
	}
	if req.Capabilities != nil {
		if err := s.UpsertCapabilities(ctx, deviceID, *req.Capabilities); err != nil {
			return resp, err
		}
	}
	s.tel.RecordWorkerMetrics(deviceID, req.Metrics)
	invalid, err := s.sched.Reconcile(ctx, projectID, deviceID, req.Attempts)
	if err != nil {
		return resp, err
	}
	if invalid != nil {
		resp.InvalidAttempts = invalid
	}
	if err := s.d.DB.QueryRow(ctx, `SELECT count(*) FROM worker_commands WHERE device_id=$1 AND state IN ('queued','delivered') AND expires_at > now()`,
		deviceID).Scan(&resp.PendingCommands); err != nil {
		return resp, err
	}
	s.publishIfChanged(ctx, projectID, deviceID, req.Metrics)
	return resp, nil
}

// UpsertCapabilities validates and stores reported capabilities.
func (s *Service) UpsertCapabilities(ctx context.Context, deviceID string, c protocol.Capabilities) error {
	if c.SchemaVersion != protocol.CapabilitiesSchemaVersion {
		return fmt.Errorf("unsupported capabilities schema %d", c.SchemaVersion)
	}
	if len(c.GPUs) > 16 || len(c.Executors) > 64 || len(c.Backends) > 16 {
		return errors.New("capabilities too large")
	}
	for _, e := range c.Executors {
		if len(e) > 64 {
			return errors.New("executor name too long")
		}
	}
	if c.RAMBudgetMB <= 0 || c.RAMBudgetMB > c.RAMMB {
		c.RAMBudgetMB = c.RAMMB
	}
	if c.VRAMBudgetMB < 0 || c.VRAMBudgetMB > c.VRAMMB {
		c.VRAMBudgetMB = c.VRAMMB
	}
	cpu, _ := json.Marshal(c.CPU)
	gpus, _ := json.Marshal(c.GPUs)
	if c.GPUs == nil {
		gpus = []byte(`[]`)
	}
	bench, _ := json.Marshal(c.Benchmark)
	if c.Benchmark == nil {
		bench = []byte(`{}`)
	}
	_, err := s.d.DB.Exec(ctx, `INSERT INTO worker_capabilities (device_id, cpu, gpus, ram_mb, vram_mb, ram_budget_mb, vram_budget_mb,
			executors, compute_backends, storage_free_mb, benchmark, schema_version, updated_at)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12, now())
		ON CONFLICT (device_id) DO UPDATE SET cpu=EXCLUDED.cpu, gpus=EXCLUDED.gpus, ram_mb=EXCLUDED.ram_mb, vram_mb=EXCLUDED.vram_mb,
			ram_budget_mb=EXCLUDED.ram_budget_mb, vram_budget_mb=EXCLUDED.vram_budget_mb, executors=EXCLUDED.executors,
			compute_backends=EXCLUDED.compute_backends, storage_free_mb=EXCLUDED.storage_free_mb, benchmark=EXCLUDED.benchmark,
			schema_version=EXCLUDED.schema_version, updated_at=now()`,
		deviceID, cpu, gpus, c.RAMMB, c.VRAMMB, c.RAMBudgetMB, c.VRAMBudgetMB, c.Executors, c.Backends, c.StorageFreeMB, bench, c.SchemaVersion)
	if err == nil {
		_, err = s.d.DB.Exec(ctx, `UPDATE devices SET os=$2, arch=$3, updated_at=now() WHERE id=$1`, deviceID, truncate(c.OS, 32), truncate(c.Arch, 32))
	}
	return err
}

// View is the dashboard representation of a worker.
type View struct {
	ID              string                    `json:"id"`
	Name            string                    `json:"name"`
	ContributorName string                    `json:"contributor_name"`
	OS              string                    `json:"os"`
	Arch            string                    `json:"arch"`
	AgentVersion    string                    `json:"agent_version"`
	State           string                    `json:"state"`
	ReportedState   string                    `json:"reported_state"`
	Paused          bool                      `json:"paused"`
	PublicProfile   bool                      `json:"public_profile"`
	LastSeenAt      *time.Time                `json:"last_seen_at"`
	CreatedAt       time.Time                 `json:"created_at"`
	ClockSkewMS     int                       `json:"clock_skew_ms"`
	CPU             protocol.CPUInfo          `json:"cpu"`
	GPUs            []protocol.GPUInfo        `json:"gpus"`
	RAMMB           int64                     `json:"ram_mb"`
	VRAMMB          int64                     `json:"vram_mb"`
	RAMBudgetMB     int64                     `json:"ram_budget_mb"`
	VRAMBudgetMB    int64                     `json:"vram_budget_mb"`
	Executors       []string                  `json:"executors"`
	CurrentTask     *Task                     `json:"current_task"`
	Metrics         *protocol.MetricsSnapshot `json:"metrics"`
	AcceptedJobs    int64                     `json:"accepted_jobs"`
	Credit          float64                   `json:"credit"`
}

// Task summarises the device's active attempt.
type Task struct {
	JobID     string  `json:"job_id"`
	AttemptID string  `json:"attempt_id"`
	JobType   string  `json:"job_type"`
	State     string  `json:"state"`
	Progress  float64 `json:"progress"`
	Stage     string  `json:"stage"`
}

const viewSelect = `SELECT d.id, d.name, d.contributor_name, d.os, d.arch, d.agent_version, d.status, d.reported_state, d.paused,
	d.public_profile, d.last_seen_at, d.created_at,
	COALESCE((SELECT clock_skew_ms FROM worker_sessions s WHERE s.device_id=d.id ORDER BY authenticated_at DESC LIMIT 1), 0),
	COALESCE(c.cpu, '{}'), COALESCE(c.gpus, '[]'), COALESCE(c.ram_mb, 0), COALESCE(c.vram_mb, 0),
	COALESCE(c.ram_budget_mb, 0), COALESCE(c.vram_budget_mb, 0), COALESCE(c.executors, '{}'),
	t.job_id::text, t.id::text, t.job_type, t.state, t.progress, t.stage,
	COALESCE(k.n, 0), COALESCE(k.credit, 0)
	FROM devices d
	LEFT JOIN worker_capabilities c ON c.device_id = d.id
	LEFT JOIN LATERAL (SELECT a.id, a.job_id, j.job_type, a.state, a.progress, a.stage FROM job_attempts a JOIN jobs j ON j.id=a.job_id
		WHERE a.device_id=d.id AND a.state IN ('leased','running','uploading') ORDER BY a.leased_at DESC LIMIT 1) t ON true
	LEFT JOIN LATERAL (SELECT count(*) AS n, sum(credit) AS credit FROM contributions x WHERE x.device_id=d.id) k ON true `

func (s *Service) scan(r pgx.Row) (View, error) {
	var v View
	var status string
	var cpu, gpus []byte
	var tJob, tAtt, tType, tState, tStage *string
	var tProg *float64
	err := r.Scan(&v.ID, &v.Name, &v.ContributorName, &v.OS, &v.Arch, &v.AgentVersion, &status, &v.ReportedState, &v.Paused,
		&v.PublicProfile, &v.LastSeenAt, &v.CreatedAt, &v.ClockSkewMS, &cpu, &gpus, &v.RAMMB, &v.VRAMMB, &v.RAMBudgetMB,
		&v.VRAMBudgetMB, &v.Executors, &tJob, &tAtt, &tType, &tState, &tProg, &tStage, &v.AcceptedJobs, &v.Credit)
	if err != nil {
		return v, err
	}
	_ = json.Unmarshal(cpu, &v.CPU)
	_ = json.Unmarshal(gpus, &v.GPUs)
	if v.GPUs == nil {
		v.GPUs = []protocol.GPUInfo{}
	}
	if tJob != nil {
		v.CurrentTask = &Task{JobID: *tJob, AttemptID: *tAtt, JobType: *tType, State: *tState, Progress: *tProg, Stage: *tStage}
	}
	v.State = ComputeState(status, v.Paused, v.LastSeenAt, v.ReportedState, v.CurrentTask != nil, s.disconnectAfter())
	if m, ok := s.tel.LatestWorkerMetrics(v.ID); ok {
		v.Metrics = &m
	}
	return v, nil
}

// List returns the project's workers, optionally filtered by state, with paging.
func (s *Service) List(ctx context.Context, projectID string, limit, offset int) ([]View, int, error) {
	if limit <= 0 || limit > 500 {
		limit = 200
	}
	var total int
	if err := s.d.DB.QueryRow(ctx, `SELECT count(*) FROM devices WHERE project_id=$1`, projectID).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.d.DB.Query(ctx, viewSelect+`WHERE d.project_id=$1 ORDER BY d.status, d.last_seen_at DESC NULLS LAST, d.name LIMIT $2 OFFSET $3`,
		projectID, limit, offset)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	out := []View{}
	for rows.Next() {
		v, err := s.scan(rows)
		if err != nil {
			return nil, 0, err
		}
		out = append(out, v)
	}
	return out, total, rows.Err()
}

// Get returns one worker.
func (s *Service) Get(ctx context.Context, projectID, deviceID string) (View, error) {
	v, err := s.scan(s.d.DB.QueryRow(ctx, viewSelect+`WHERE d.project_id=$1 AND d.id=$2`, projectID, deviceID))
	if errors.Is(err, pgx.ErrNoRows) {
		return v, ErrNotFound
	}
	return v, err
}

// Counts returns online/total and GPU counts for the overview.
func (s *Service) Counts(ctx context.Context, projectID string) (online, total, activeGPUs int, err error) {
	err = s.d.DB.QueryRow(ctx, `SELECT
			count(*) FILTER (WHERE d.status='active' AND d.last_seen_at > now() - make_interval(secs => $2)),
			count(*),
			COALESCE(sum(jsonb_array_length(c.gpus)) FILTER (WHERE d.status='active' AND d.last_seen_at > now() - make_interval(secs => $2)), 0)
		FROM devices d LEFT JOIN worker_capabilities c ON c.device_id=d.id WHERE d.project_id=$1`,
		projectID, s.disconnectAfter().Seconds()).Scan(&online, &total, &activeGPUs)
	return
}

// Rename changes a device's display name.
func (s *Service) Rename(ctx context.Context, actor core.Actor, projectID, deviceID, name string) error {
	if name == "" || len(name) > 64 {
		return errors.New("name must be 1-64 characters")
	}
	tag, err := s.d.DB.Exec(ctx, `UPDATE devices SET name=$3, updated_at=now() WHERE id=$2 AND project_id=$1`, projectID, deviceID, name)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return ErrNotFound
	}
	s.d.Audit(ctx, nil, actor, "worker.rename", "device:"+deviceID, "success", map[string]any{"name": name})
	s.publish(ctx, projectID, deviceID)
	return nil
}

// SetPublicProfile toggles leaderboard visibility.
func (s *Service) SetPublicProfile(ctx context.Context, projectID, deviceID string, public bool) error {
	_, err := s.d.DB.Exec(ctx, `UPDATE devices SET public_profile=$3 WHERE id=$2 AND project_id=$1`, projectID, deviceID, public)
	return err
}

// Revoke permanently disables a device: sessions end, active attempts are
// cancelled and requeued, and the key can never re-enroll in this project.
func (s *Service) Revoke(ctx context.Context, actor core.Actor, projectID, deviceID string) error {
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		tag, err := tx.Exec(ctx, `UPDATE devices SET status='revoked', revoked_at=now(), updated_at=now() WHERE id=$2 AND project_id=$1 AND status='active'`,
			projectID, deviceID)
		if err != nil {
			return err
		}
		if tag.RowsAffected() == 0 {
			return ErrNotFound
		}
		if _, err := tx.Exec(ctx, `UPDATE worker_sessions SET revoked_at=now(), connection_state='revoked' WHERE device_id=$1 AND revoked_at IS NULL`, deviceID); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE jobs j SET status=CASE WHEN j.attempt_count < j.max_attempts THEN 'ready' ELSE 'failed' END,
				current_attempt_id=NULL, failure_reason='worker revoked', updated_at=now()
			FROM job_attempts a WHERE a.id=j.current_attempt_id AND a.device_id=$1 AND a.state IN ('leased','running','uploading')`, deviceID); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE job_attempts SET state='cancelled', completed_at=now(), error='worker revoked'
			WHERE device_id=$1 AND state IN ('leased','running','uploading')`, deviceID); err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `UPDATE worker_commands SET state='expired', finished_at=now() WHERE device_id=$1 AND state IN ('queued','delivered','acknowledged','executing')`, deviceID)
		return err
	})
	if err != nil {
		return err
	}
	s.d.Audit(ctx, nil, actor, "worker.revoke", "device:"+deviceID, "success", nil)
	s.d.Events.NotifyJobs(projectID)
	s.publish(ctx, projectID, deviceID)
	return nil
}

// CommandView is the dashboard representation of a command.
type CommandView struct {
	ID             string          `json:"id"`
	DeviceID       string          `json:"device_id"`
	Type           string          `json:"type"`
	State          string          `json:"state"`
	Payload        json.RawMessage `json:"payload"`
	Result         json.RawMessage `json:"result"`
	CreatedAt      time.Time       `json:"created_at"`
	DeliveredAt    *time.Time      `json:"delivered_at"`
	AcknowledgedAt *time.Time      `json:"acknowledged_at"`
	FinishedAt     *time.Time      `json:"finished_at"`
	ExpiresAt      time.Time       `json:"expires_at"`
}

// SendCommand queues a reliable command. pause/resume also update the
// authoritative scheduler flag immediately, so no new work is leased even
// before the worker confirms.
func (s *Service) SendCommand(ctx context.Context, actor core.Actor, projectID, deviceID, cmdType string, payload json.RawMessage) (CommandView, error) {
	switch cmdType {
	case protocol.CmdPause, protocol.CmdResume, protocol.CmdStopTask, protocol.CmdDisconnect, protocol.CmdRefreshCapabilities:
	default:
		return CommandView{}, ErrBadCommand
	}
	if len(payload) == 0 {
		payload = json.RawMessage(`{}`)
	}
	if len(payload) > 4096 || !json.Valid(payload) {
		return CommandView{}, ErrBadCommand
	}
	var by any
	if actor.Type == "user" {
		by = actor.ID
	}
	var c CommandView
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var status string
		if err := tx.QueryRow(ctx, `SELECT status FROM devices WHERE id=$1 AND project_id=$2 FOR UPDATE`, deviceID, projectID).Scan(&status); err != nil {
			return ErrNotFound
		}
		if status != "active" {
			return ErrNotFound
		}
		switch cmdType {
		case protocol.CmdPause:
			if _, err := tx.Exec(ctx, `UPDATE devices SET paused=true, updated_at=now() WHERE id=$1`, deviceID); err != nil {
				return err
			}
		case protocol.CmdResume:
			if _, err := tx.Exec(ctx, `UPDATE devices SET paused=false, updated_at=now() WHERE id=$1`, deviceID); err != nil {
				return err
			}
		}
		return tx.QueryRow(ctx, `INSERT INTO worker_commands (device_id, command_type, payload, requested_by, expires_at)
			VALUES ($1, $2, $3, $4, now() + interval '10 minutes')
			RETURNING id, device_id, command_type, state, payload, result, created_at, delivered_at, acknowledged_at, finished_at, expires_at`,
			deviceID, cmdType, payload, by).Scan(&c.ID, &c.DeviceID, &c.Type, &c.State, &c.Payload, &c.Result, &c.CreatedAt,
			&c.DeliveredAt, &c.AcknowledgedAt, &c.FinishedAt, &c.ExpiresAt)
	})
	if err != nil {
		return c, err
	}
	if cmdType == protocol.CmdResume {
		s.d.Events.NotifyJobs(projectID)
	}
	s.d.Audit(ctx, nil, actor, "worker.command."+cmdType, "device:"+deviceID, "success", map[string]any{"command_id": c.ID})
	s.d.Events.Publish(events.Event{Type: "command.updated", ProjectID: projectID, Data: c})
	s.publish(ctx, projectID, deviceID)
	return c, nil
}

// PendingCommands returns undelivered/unfinished commands and marks queued ones delivered.
func (s *Service) PendingCommands(ctx context.Context, projectID, deviceID string) ([]protocol.Command, error) {
	rows, err := s.d.DB.Query(ctx, `UPDATE worker_commands SET state = CASE WHEN state='queued' THEN 'delivered' ELSE state END,
			delivered_at = COALESCE(delivered_at, now())
		WHERE device_id=$1 AND state IN ('queued','delivered','acknowledged','executing') AND expires_at > now()
		RETURNING id, command_type, payload, state, created_at, expires_at`, deviceID)
	if err != nil {
		return nil, err
	}
	out, err := pgx.CollectRows(rows, func(r pgx.CollectableRow) (protocol.Command, error) {
		var c protocol.Command
		err := r.Scan(&c.ID, &c.Type, &c.Payload, &c.State, &c.CreatedAt, &c.ExpiresAt)
		return c, err
	})
	if err != nil {
		return nil, err
	}
	for _, c := range out {
		s.publishCommand(ctx, projectID, c.ID)
	}
	return out, nil
}

var cmdOrder = map[string]int{protocol.CmdQueued: 0, protocol.CmdDelivered: 1, protocol.CmdAcknowledged: 2, protocol.CmdExecuting: 3,
	protocol.CmdCompleted: 4, protocol.CmdFailed: 4}

// AckCommand advances a command's state. Moving backwards is rejected;
// repeating the current state is idempotent.
func (s *Service) AckCommand(ctx context.Context, projectID, deviceID, cmdID string, req protocol.CommandAckRequest) error {
	next, ok := cmdOrder[req.State]
	if !ok || next < 2 {
		return ErrBadTransition
	}
	if len(req.Result) == 0 {
		req.Result = json.RawMessage(`{}`)
	}
	if len(req.Result) > 4096 || !json.Valid(req.Result) {
		return ErrBadCommand
	}
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var cur string
		if err := tx.QueryRow(ctx, `SELECT state FROM worker_commands WHERE id=$1 AND device_id=$2 FOR UPDATE`, cmdID, deviceID).Scan(&cur); err != nil {
			return ErrNotFound
		}
		if cur == req.State {
			return nil
		}
		if cur == protocol.CmdExpired || cmdOrder[cur] >= 4 || cmdOrder[cur] > next {
			return ErrBadTransition
		}
		_, err := tx.Exec(ctx, `UPDATE worker_commands SET state=$2, result=$3,
				acknowledged_at = COALESCE(acknowledged_at, now()),
				finished_at = CASE WHEN $2 IN ('completed','failed') THEN now() ELSE finished_at END
			WHERE id=$1`, cmdID, req.State, req.Result)
		return err
	})
	if err != nil {
		return err
	}
	s.publishCommand(ctx, projectID, cmdID)
	return nil
}

// Commands lists recent commands for a device.
func (s *Service) Commands(ctx context.Context, projectID, deviceID string, limit int) ([]CommandView, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.d.DB.Query(ctx, `SELECT c.id, c.device_id, c.command_type, c.state, c.payload, c.result, c.created_at, c.delivered_at,
			c.acknowledged_at, c.finished_at, c.expires_at
		FROM worker_commands c JOIN devices d ON d.id=c.device_id
		WHERE d.project_id=$1 AND ($2 = '' OR c.device_id::text = $2) ORDER BY c.created_at DESC LIMIT $3`, projectID, deviceID, limit)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(r pgx.CollectableRow) (CommandView, error) {
		var c CommandView
		err := r.Scan(&c.ID, &c.DeviceID, &c.Type, &c.State, &c.Payload, &c.Result, &c.CreatedAt, &c.DeliveredAt,
			&c.AcknowledgedAt, &c.FinishedAt, &c.ExpiresAt)
		return c, err
	})
}

func (s *Service) publishCommand(ctx context.Context, projectID, cmdID string) {
	if s.d.Events.Subscribers() == 0 {
		return
	}
	var c CommandView
	if err := s.d.DB.QueryRow(ctx, `SELECT id, device_id, command_type, state, payload, result, created_at, delivered_at,
			acknowledged_at, finished_at, expires_at FROM worker_commands WHERE id=$1`, cmdID).Scan(&c.ID, &c.DeviceID, &c.Type,
		&c.State, &c.Payload, &c.Result, &c.CreatedAt, &c.DeliveredAt, &c.AcknowledgedAt, &c.FinishedAt, &c.ExpiresAt); err == nil {
		s.d.Events.Publish(events.Event{Type: "command.updated", ProjectID: projectID, Data: c})
	}
}

func (s *Service) publish(ctx context.Context, projectID, deviceID string) {
	v, err := s.Get(ctx, projectID, deviceID)
	if err != nil {
		return
	}
	s.mu.Lock()
	s.lastSeen[deviceID] = v.State
	s.mu.Unlock()
	if s.d.Events.Subscribers() > 0 {
		s.d.Events.Publish(events.Event{Type: "worker.updated", ProjectID: projectID, Data: v})
	}
}

// publishIfChanged sends the full view on a state change, otherwise a light metrics event.
func (s *Service) publishIfChanged(ctx context.Context, projectID, deviceID string, m protocol.MetricsSnapshot) {
	if s.d.Events.Subscribers() == 0 {
		return
	}
	v, err := s.Get(ctx, projectID, deviceID)
	if err != nil {
		return
	}
	s.mu.Lock()
	prev := s.lastSeen[deviceID]
	s.lastSeen[deviceID] = v.State
	s.mu.Unlock()
	if prev != v.State || v.CurrentTask != nil {
		s.d.Events.Publish(events.Event{Type: "worker.updated", ProjectID: projectID, Data: v})
		return
	}
	s.d.Events.Publish(events.Event{Type: "worker.metrics", ProjectID: projectID, Data: map[string]any{"id": deviceID, "metrics": m, "last_seen_at": v.LastSeenAt}})
}

// RunPresence publishes transitions to "disconnected" that happen without
// any request (a worker simply stops heartbeating).
func (s *Service) RunPresence(ctx context.Context) {
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		rows, err := s.d.DB.Query(ctx, `SELECT id, project_id FROM devices WHERE status='active' AND last_seen_at IS NOT NULL
			AND last_seen_at < now() - make_interval(secs => $1) AND last_seen_at > now() - make_interval(secs => $1) - interval '1 hour'`,
			s.disconnectAfter().Seconds())
		if err != nil {
			continue
		}
		type pair struct{ dev, proj string }
		var ps []pair
		for rows.Next() {
			var p pair
			if rows.Scan(&p.dev, &p.proj) == nil {
				ps = append(ps, p)
			}
		}
		rows.Close()
		for _, p := range ps {
			s.mu.Lock()
			prev := s.lastSeen[p.dev]
			s.mu.Unlock()
			if prev != StateDisconnected {
				_, _ = s.d.DB.Exec(ctx, `UPDATE worker_sessions SET connection_state='disconnected' WHERE device_id=$1 AND revoked_at IS NULL`, p.dev)
				s.publish(ctx, p.proj, p.dev)
			}
		}
	}
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
