// Package telemetry batches worker metrics and logs into PostgreSQL with
// bounded buffers and retention, so telemetry cannot exhaust the controller.
package telemetry

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"nivra/internal/shared/protocol"
)

// Buffer limits; when full, the oldest data is dropped and counted.
const (
	maxBufferedMetrics = 5000
	maxBufferedLogs    = 10000
	flushEvery         = 3 * time.Second
)

// LogRecord is one stored log line.
type LogRecord struct {
	ID        int64             `json:"id"`
	Time      time.Time         `json:"ts"`
	ProjectID *string           `json:"project_id"`
	Source    string            `json:"source"`
	DeviceID  *string           `json:"device_id"`
	Service   string            `json:"service"`
	RunID     *string           `json:"run_id"`
	Level     string            `json:"level"`
	Message   string            `json:"message"`
	Fields    map[string]string `json:"fields"`
}

type metricRow struct {
	device string
	ts     time.Time
	m      protocol.MetricsSnapshot
}

// Service is the telemetry pipeline.
type Service struct {
	db *pgxpool.Pool

	mu      sync.Mutex
	metrics []metricRow
	logs    []LogRecord
	latest  map[string]protocol.MetricsSnapshot
	dropped int64

	retainMetrics, retainLogs time.Duration
	onLog                     func(LogRecord)
}

// New creates the telemetry service.
func New(db *pgxpool.Pool, metricsDays, logDays int) *Service {
	return &Service{db: db, latest: map[string]protocol.MetricsSnapshot{},
		retainMetrics: time.Duration(metricsDays) * 24 * time.Hour, retainLogs: time.Duration(logDays) * 24 * time.Hour}
}

// OnLog registers a callback for live log streaming.
func (s *Service) OnLog(f func(LogRecord)) { s.onLog = f }

// RecordWorkerMetrics buffers a worker metrics sample.
func (s *Service) RecordWorkerMetrics(device string, m protocol.MetricsSnapshot) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.latest[device] = m
	if len(s.metrics) >= maxBufferedMetrics {
		s.metrics = s.metrics[1:]
		s.dropped++
	}
	s.metrics = append(s.metrics, metricRow{device, time.Now().UTC(), m})
}

// LatestWorkerMetrics returns the most recent in-memory sample.
func (s *Service) LatestWorkerMetrics(device string) (protocol.MetricsSnapshot, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m, ok := s.latest[device]
	return m, ok
}

// Log buffers a log record.
func (s *Service) Log(r LogRecord) {
	if r.Time.IsZero() {
		r.Time = time.Now().UTC()
	}
	switch r.Level {
	case "debug", "info", "warn", "error":
	default:
		r.Level = "info"
	}
	if len(r.Message) > 4000 {
		r.Message = r.Message[:4000]
	}
	s.mu.Lock()
	if len(s.logs) >= maxBufferedLogs {
		s.logs = s.logs[1:]
		s.dropped++
	}
	s.logs = append(s.logs, r)
	cb := s.onLog
	s.mu.Unlock()
	if cb != nil {
		cb(r)
	}
}

// WorkerLogs stores a batch shipped by a worker.
func (s *Service) WorkerLogs(projectID, deviceID string, b protocol.LogBatch) {
	if len(b.Lines) > 500 {
		b.Lines = b.Lines[len(b.Lines)-500:]
	}
	for _, l := range b.Lines {
		p, d := projectID, deviceID
		ts := time.UnixMilli(l.TimeMS).UTC()
		if l.TimeMS == 0 || time.Since(ts) > 7*24*time.Hour || time.Until(ts) > time.Hour {
			ts = time.Now().UTC() // never trust unvalidated worker clocks for ordering
		}
		s.Log(LogRecord{Time: ts, ProjectID: &p, Source: "worker", DeviceID: &d, Service: truncate(l.Service, 64),
			Level: strings.ToLower(l.Level), Message: l.Message, Fields: limitFields(l.Fields)})
	}
}

func limitFields(f map[string]string) map[string]string {
	if len(f) == 0 {
		return nil
	}
	out := map[string]string{}
	for k, v := range f {
		if len(out) >= 20 {
			break
		}
		out[truncate(k, 64)] = truncate(v, 512)
	}
	return out
}

// Dropped returns how many telemetry records were dropped due to backpressure.
func (s *Service) Dropped() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dropped
}

// Flush writes buffered telemetry with COPY.
func (s *Service) Flush(ctx context.Context) error {
	s.mu.Lock()
	ms, ls := s.metrics, s.logs
	s.metrics, s.logs = nil, nil
	s.mu.Unlock()
	if len(ms) > 0 {
		rows := make([][]any, len(ms))
		for i, r := range ms {
			rows[i] = []any{r.device, r.ts, float32(r.m.CPUPct), int32(r.m.RAMUsedMB), float32(r.m.GPUPct), int32(r.m.VRAMUsedMB),
				float32(r.m.NetRxKbps), float32(r.m.NetTxKbps)}
		}
		if _, err := s.db.CopyFrom(ctx, pgx.Identifier{"worker_metrics"},
			[]string{"device_id", "ts", "cpu_pct", "ram_used_mb", "gpu_pct", "vram_used_mb", "net_rx_kbps", "net_tx_kbps"},
			pgx.CopyFromRows(rows)); err != nil {
			return err
		}
	}
	if len(ls) > 0 {
		rows := make([][]any, len(ls))
		for i, r := range ls {
			f, _ := json.Marshal(r.Fields)
			if r.Fields == nil {
				f = []byte(`{}`)
			}
			rows[i] = []any{r.Time, r.ProjectID, r.Source, r.DeviceID, r.Service, r.RunID, r.Level, r.Message, f}
		}
		if _, err := s.db.CopyFrom(ctx, pgx.Identifier{"log_events"},
			[]string{"ts", "project_id", "source", "device_id", "service", "run_id", "level", "message", "fields"},
			pgx.CopyFromRows(rows)); err != nil {
			return err
		}
	}
	return nil
}

// Run flushes periodically and applies retention hourly until ctx ends,
// then performs a final flush.
func (s *Service) Run(ctx context.Context, log *slog.Logger) {
	t := time.NewTicker(flushEvery)
	defer t.Stop()
	lastRetention := time.Time{}
	for {
		select {
		case <-ctx.Done():
			fctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			if err := s.Flush(fctx); err != nil {
				log.Warn("final telemetry flush", "err", err)
			}
			cancel()
			return
		case <-t.C:
		}
		if err := s.Flush(ctx); err != nil && ctx.Err() == nil {
			log.Warn("telemetry flush", "err", err)
		}
		if time.Since(lastRetention) > time.Hour {
			lastRetention = time.Now()
			if err := s.ApplyRetention(ctx); err != nil && ctx.Err() == nil {
				log.Warn("telemetry retention", "err", err)
			}
		}
	}
}

// ApplyRetention deletes telemetry older than the configured windows.
func (s *Service) ApplyRetention(ctx context.Context) error {
	if _, err := s.db.Exec(ctx, `DELETE FROM worker_metrics WHERE ts < now() - make_interval(secs => $1)`, s.retainMetrics.Seconds()); err != nil {
		return err
	}
	if _, err := s.db.Exec(ctx, `DELETE FROM log_events WHERE ts < now() - make_interval(secs => $1)`, s.retainLogs.Seconds()); err != nil {
		return err
	}
	_, err := s.db.Exec(ctx, `DELETE FROM audit_events WHERE ts < now() - interval '365 days'`)
	return err
}

// LogQuery filters stored logs.
type LogQuery struct {
	ProjectID string
	DeviceID  string
	Service   string
	RunID     string
	MinLevel  string
	Search    string
	Since     *time.Time
	Until     *time.Time
	BeforeID  int64
	Limit     int
}

var levelRank = map[string]int{"debug": 0, "info": 1, "warn": 2, "error": 3}

// QueryLogs returns logs newest first.
func (s *Service) QueryLogs(ctx context.Context, q LogQuery) ([]LogRecord, error) {
	if q.Limit <= 0 || q.Limit > 1000 {
		q.Limit = 200
	}
	levels := []string{}
	for l, r := range levelRank {
		if r >= levelRank[q.MinLevel] {
			levels = append(levels, l)
		}
	}
	rows, err := s.db.Query(ctx, `SELECT id, ts, project_id::text, source, device_id::text, service, run_id::text, level, message, fields
		FROM log_events
		WHERE (project_id IS NULL OR project_id::text = $1)
		  AND ($2 = '' OR device_id::text = $2) AND ($3 = '' OR service = $3) AND ($4 = '' OR run_id::text = $4)
		  AND level = ANY($5) AND ($6 = '' OR message ILIKE '%' || $6 || '%')
		  AND ($7::timestamptz IS NULL OR ts >= $7) AND ($8::timestamptz IS NULL OR ts <= $8)
		  AND ($9 = 0 OR id < $9)
		ORDER BY id DESC LIMIT $10`, q.ProjectID, q.DeviceID, q.Service, q.RunID, levels, escapeLike(q.Search), q.Since, q.Until, q.BeforeID, q.Limit)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(r pgx.CollectableRow) (LogRecord, error) {
		var l LogRecord
		var f []byte
		err := r.Scan(&l.ID, &l.Time, &l.ProjectID, &l.Source, &l.DeviceID, &l.Service, &l.RunID, &l.Level, &l.Message, &f)
		_ = json.Unmarshal(f, &l.Fields)
		return l, err
	})
}

func escapeLike(s string) string {
	return strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`).Replace(s)
}

// MetricPoint is a time-series sample.
type MetricPoint struct {
	Time time.Time `json:"t"`
	CPU  float64   `json:"cpu"`
	RAM  int64     `json:"ram_mb"`
	GPU  float64   `json:"gpu"`
	VRAM int64     `json:"vram_mb"`
}

// WorkerHistory returns metrics for one device, downsampled to <= 300 points.
func (s *Service) WorkerHistory(ctx context.Context, deviceID string, since time.Duration) ([]MetricPoint, error) {
	bucket := since.Seconds() / 300
	if bucket < 1 {
		bucket = 1
	}
	rows, err := s.db.Query(ctx, `SELECT to_timestamp(floor(extract(epoch FROM ts) / $3) * $3) AS b,
			avg(cpu_pct), avg(ram_used_mb)::bigint, avg(gpu_pct), avg(vram_used_mb)::bigint
		FROM worker_metrics WHERE device_id=$1 AND ts > now() - make_interval(secs => $2)
		GROUP BY b ORDER BY b`, deviceID, since.Seconds(), bucket)
	if err != nil {
		return nil, err
	}
	return pgx.CollectRows(rows, func(r pgx.CollectableRow) (MetricPoint, error) {
		var p MetricPoint
		err := r.Scan(&p.Time, &p.CPU, &p.RAM, &p.GPU, &p.VRAM)
		return p, err
	})
}

func isUUID(s string) bool {
	if len(s) != 36 {
		return false
	}
	for i, c := range s {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if c != '-' {
				return false
			}
		} else if !strings.ContainsRune("0123456789abcdefABCDEF", c) {
			return false
		}
	}
	return true
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// Handler is a slog.Handler that tees controller logs into the log store.
type Handler struct {
	next    slog.Handler
	tel     *Service
	service string
	attrs   []slog.Attr
}

// NewHandler wraps next.
func NewHandler(next slog.Handler, tel *Service, service string) *Handler {
	return &Handler{next: next, tel: tel, service: service}
}

// Enabled implements slog.Handler.
func (h *Handler) Enabled(ctx context.Context, l slog.Level) bool { return h.next.Enabled(ctx, l) }

// Handle implements slog.Handler.
func (h *Handler) Handle(ctx context.Context, r slog.Record) error {
	if r.Level >= slog.LevelInfo {
		fields := map[string]string{}
		var proj, run *string
		add := func(a slog.Attr) bool {
			v := a.Value.String()
			switch a.Key {
			case "project_id":
				if isUUID(v) {
					proj = &v
				}
			case "run_id":
				if isUUID(v) {
					run = &v
				}
			}
			if len(fields) < 20 {
				fields[a.Key] = truncate(v, 512)
			}
			return true
		}
		for _, a := range h.attrs {
			add(a)
		}
		r.Attrs(add)
		lvl := "info"
		switch {
		case r.Level >= slog.LevelError:
			lvl = "error"
		case r.Level >= slog.LevelWarn:
			lvl = "warn"
		}
		h.tel.Log(LogRecord{Time: r.Time.UTC(), Source: "controller", Service: h.service, Level: lvl, Message: r.Message,
			Fields: fields, ProjectID: proj, RunID: run})
	}
	return h.next.Handle(ctx, r)
}

// WithAttrs implements slog.Handler.
func (h *Handler) WithAttrs(as []slog.Attr) slog.Handler {
	return &Handler{next: h.next.WithAttrs(as), tel: h.tel, service: h.service, attrs: append(append([]slog.Attr{}, h.attrs...), as...)}
}

// WithGroup implements slog.Handler.
func (h *Handler) WithGroup(name string) slog.Handler {
	return &Handler{next: h.next.WithGroup(name), tel: h.tel, service: h.service, attrs: h.attrs}
}
