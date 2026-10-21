package agent

import (
	"sync"
	"time"

	"nivra/internal/shared/protocol"
)

// LogLine is a local log entry.
type LogLine struct {
	Seq     int64     `json:"seq"`
	Time    time.Time `json:"time"`
	Level   string    `json:"level"`
	Service string    `json:"service"`
	Message string    `json:"message"`
}

// logRing keeps recent lines for the local UI and ships them to the controller.
type logRing struct {
	mu      sync.Mutex
	lines   []LogLine
	seq     int64
	shipped int64
	max     int
}

func newLogRing(n int) *logRing { return &logRing{max: n} }

func (r *logRing) add(level, service, msg string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seq++
	r.lines = append(r.lines, LogLine{Seq: r.seq, Time: time.Now(), Level: level, Service: service, Message: msg})
	if len(r.lines) > r.max {
		r.lines = r.lines[len(r.lines)-r.max:]
	}
}

// since returns lines with Seq > after.
func (r *logRing) since(after int64, limit int) []LogLine {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []LogLine
	for _, l := range r.lines {
		if l.Seq > after {
			out = append(out, l)
		}
	}
	if len(out) > limit {
		out = out[len(out)-limit:]
	}
	return out
}

// unshipped returns info+ lines not yet shipped, up to limit.
func (r *logRing) unshipped(limit int) ([]protocol.LogLine, int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []protocol.LogLine
	last := r.shipped
	for _, l := range r.lines {
		if l.Seq <= r.shipped || l.Level == "debug" {
			continue
		}
		if len(out) >= limit {
			break
		}
		out = append(out, protocol.LogLine{TimeMS: l.Time.UnixMilli(), Level: l.Level, Service: l.Service, Message: l.Message})
		last = l.Seq
	}
	if len(out) == 0 {
		last = r.seq
	}
	return out, last
}

func (r *logRing) markShipped(seq int64) {
	r.mu.Lock()
	if seq > r.shipped {
		r.shipped = seq
	}
	r.mu.Unlock()
}
