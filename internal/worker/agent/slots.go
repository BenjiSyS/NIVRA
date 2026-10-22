package agent

import (
	"context"
	"fmt"
	"path/filepath"

	"nivra/internal/shared/config"
)

// A computer with several CUDA GPUs runs one worker slot per GPU in the same
// process (docs/architecture/LONG_VIDEO.md). Slot 0 is the normal worker;
// slot k has its own identity and data folder <data_dir>/gpu<k> and is pinned
// to GPU k, so the scheduler (one lease per device) gives every GPU its own
// job, e.g. one segment of a long video each.

// NewSlots creates the primary agent plus one extra agent per additional
// CUDA GPU. CPU-only computers get just the primary.
func NewSlots(ctx context.Context, cfg config.Worker) ([]*Agent, error) {
	primary, err := New(ctx, cfg)
	if err != nil {
		return nil, err
	}
	slots := []*Agent{primary}
	primary.mu.Lock()
	gpus, cuda := len(primary.hw.GPUs), primary.hw.AI.CUDA
	base := primary.settings
	primary.mu.Unlock()
	if !cuda || gpus < 2 {
		return slots, nil
	}
	for k := 1; k < gpus; k++ {
		c := cfg
		c.DataDir = filepath.Join(cfg.DataDir, fmt.Sprintf("gpu%d", k))
		a, err := New(ctx, c)
		if err != nil {
			return nil, fmt.Errorf("worker slot for GPU %d: %w", k, err)
		}
		a.slot = k
		a.gate = primary.gate
		a.gate.register(a)
		a.mu.Lock()
		s := a.settings
		a.mu.Unlock()
		// Pin the slot to its GPU; share the CPU budget so N slots do not
		// oversubscribe the machine. RAM stays per slot (each loads a model).
		want := s
		want.GPUIndex = k
		want.CPUThreads = max(1, base.CPUThreads/gpus)
		want.AutoStart = base.AutoStart
		want.Schedule = base.Schedule
		if want != s {
			if err := a.UpdateSettings(want); err != nil {
				return nil, fmt.Errorf("worker slot for GPU %d: %w", k, err)
			}
		}
		slots = append(slots, a)
	}
	if base.GPUIndex != 0 && base.GPUIndex != -1 {
		// The primary was pointed at another GPU; keep slots distinct.
		primary.logf("warn", "agent", "primary worker uses GPU %d; slot for that GPU duplicates it", base.GPUIndex)
	}
	return slots, nil
}

// Slot is 0 for the primary worker, k for the worker pinned to GPU k.
func (a *Agent) Slot() int { return a.slot }
