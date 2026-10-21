// Package events fans controller state changes out to dashboard SSE clients
// and wakes long-polling workers when new jobs become ready.
package events

import (
	"encoding/json"
	"sync"
	"time"
)

// Event is one real-time update. Data must be JSON-serialisable and must not
// contain secrets: every authorised dashboard user of the project sees it.
type Event struct {
	Type      string    `json:"type"`
	ProjectID string    `json:"project_id,omitempty"`
	Time      time.Time `json:"time"`
	Data      any       `json:"data,omitempty"`
}

// Encoded is a pre-serialised event ready to write to an SSE stream.
type Encoded struct {
	Type string
	JSON []byte
}

type subscriber struct {
	project string
	ch      chan Encoded
	lagged  bool
}

// Hub is a bounded, non-blocking pub/sub. Slow subscribers are dropped and
// told to resynchronise instead of blocking publishers (backpressure).
type Hub struct {
	mu   sync.Mutex
	subs map[*subscriber]struct{}

	jobMu   sync.Mutex
	jobWake map[string]chan struct{}
}

// NewHub creates an empty hub.
func NewHub() *Hub {
	return &Hub{subs: map[*subscriber]struct{}{}, jobWake: map[string]chan struct{}{}}
}

// SubscriberBuffer is the per-client queue length before it is considered lagging.
const SubscriberBuffer = 512

// Subscribe registers a client for one project ("" = all projects). The
// returned channel is closed when the client lags or cancel is called.
func (h *Hub) Subscribe(projectID string) (<-chan Encoded, func()) {
	s := &subscriber{project: projectID, ch: make(chan Encoded, SubscriberBuffer)}
	h.mu.Lock()
	h.subs[s] = struct{}{}
	h.mu.Unlock()
	var once sync.Once
	return s.ch, func() {
		once.Do(func() {
			h.mu.Lock()
			if _, ok := h.subs[s]; ok {
				delete(h.subs, s)
				close(s.ch)
			}
			h.mu.Unlock()
		})
	}
}

// Publish sends an event to all matching subscribers without blocking.
func (h *Hub) Publish(e Event) {
	if e.Time.IsZero() {
		e.Time = time.Now().UTC()
	}
	b, err := json.Marshal(e)
	if err != nil {
		return
	}
	enc := Encoded{Type: e.Type, JSON: b}
	h.mu.Lock()
	defer h.mu.Unlock()
	for s := range h.subs {
		if s.project != "" && e.ProjectID != "" && s.project != e.ProjectID {
			continue
		}
		select {
		case s.ch <- enc:
		default:
			// Client cannot keep up: drop it; the browser reconnects and refetches.
			s.lagged = true
			delete(h.subs, s)
			close(s.ch)
		}
	}
}

// Subscribers returns the number of connected clients.
func (h *Hub) Subscribers() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.subs)
}

// JobWait returns a channel closed the next time NotifyJobs is called for the project.
func (h *Hub) JobWait(projectID string) <-chan struct{} {
	h.jobMu.Lock()
	defer h.jobMu.Unlock()
	ch, ok := h.jobWake[projectID]
	if !ok {
		ch = make(chan struct{})
		h.jobWake[projectID] = ch
	}
	return ch
}

// NotifyJobs wakes every worker long-polling for jobs in the project.
func (h *Hub) NotifyJobs(projectID string) {
	h.jobMu.Lock()
	defer h.jobMu.Unlock()
	if ch, ok := h.jobWake[projectID]; ok {
		close(ch)
		delete(h.jobWake, projectID)
	}
}
