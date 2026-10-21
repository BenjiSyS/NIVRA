package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Verdict is a validator's decision about a submitted result.
type Verdict struct {
	Accept bool
	Reason string
	// Units / UnitKind quantify verified work (e.g. 5e6 "sha256_iterations").
	Units    float64
	UnitKind string
	// Credit is the server-computed contribution credit for an accepted result.
	Credit float64
	Detail map[string]any
	// Retry asks the scheduler to requeue the job after a rejection.
	Retry bool
}

// ValidationInput gives validators everything they need; it is read-only.
type ValidationInput struct {
	JobID        string
	AttemptID    string
	ResultID     string
	ProjectID    string
	DeviceID     string
	JobType      string
	Params       json.RawMessage
	Result       json.RawMessage
	ArtifactID   *string
	Artifacts    map[string]string
	ModelVersion *string
	TrainingRun  *string
	Duration     time.Duration
}

// JobType describes a schedulable job type.
type JobType struct {
	Name     string
	Category string
	// ValidateParams rejects malformed params at job creation.
	ValidateParams func(json.RawMessage) error
	// Validate decides whether a result is accepted. It must be idempotent:
	// after a controller crash it may run again for the same result.
	Validate func(context.Context, ValidationInput) (Verdict, error)
	// Async validators run on the background validation pool instead of
	// inside the worker's completion request.
	Async bool
	// OnFinalized runs after the verdict commits (e.g. training bookkeeping).
	OnFinalized func(context.Context, ValidationInput, Verdict)
}

// Registry maps job type names to definitions.
type Registry struct {
	mu    sync.RWMutex
	types map[string]*JobType
}

// NewRegistry creates an empty registry.
func NewRegistry() *Registry { return &Registry{types: map[string]*JobType{}} }

// Register adds a job type; duplicate names panic (programming error).
func (r *Registry) Register(t *JobType) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, dup := r.types[t.Name]; dup {
		panic(fmt.Sprintf("scheduler: job type %q registered twice", t.Name))
	}
	if t.Validate == nil || t.ValidateParams == nil {
		panic(fmt.Sprintf("scheduler: job type %q needs ValidateParams and Validate", t.Name))
	}
	r.types[t.Name] = t
}

// Get returns a job type.
func (r *Registry) Get(name string) (*JobType, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.types[name]
	return t, ok
}

// Names lists registered job types.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.types))
	for n := range r.types {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
