package agent

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Gate keeps the Nivra network worker and local generations in the Nivra App
// from using the same GPU at once (two models in 6 GB of VRAM means an
// out-of-memory failure). While a local generation runs, no new network task is
// leased; while a network task runs, a local generation is refused with a plain
// message. One Gate is shared by all worker slots of the process.
type Gate struct {
	mu     sync.Mutex // serialises "a task starts" against "a local run starts"
	agents []*Agent
	local  atomic.Int32
}

// NewGate returns an empty gate; agents register themselves.
func NewGate() *Gate { return &Gate{} }

func (g *Gate) register(a *Agent) {
	g.mu.Lock()
	g.agents = append(g.agents, a)
	g.mu.Unlock()
}

// LocalActive reports whether a local generation holds the gate. It never blocks.
func (g *Gate) LocalActive() bool { return g != nil && g.local.Load() > 0 }

// BusyError says the network worker is using this computer.
type BusyError struct {
	// ETA is the estimated time until the network task ends (0 = unknown).
	ETA time.Duration
}

func (e *BusyError) Error() string {
	when := "in a few minutes"
	if e.ETA > 0 {
		m := int((e.ETA + 59*time.Second) / time.Minute)
		when = fmt.Sprintf("in about %d min", max(m, 1))
	}
	return "Your computer is busy helping the Nivra network on this GPU. It will finish " + when +
		", or pause sharing in Settings."
}

// busyLocked returns a BusyError when any agent is running a task. g.mu is held.
func (g *Gate) busyLocked() error {
	var best *BusyError
	for _, a := range g.agents {
		t := a.currentTask()
		if t == nil {
			continue
		}
		v := t.view()
		var eta time.Duration
		if v.Progress >= 0.03 && v.Progress < 1 {
			eta = time.Duration(v.ElapsedS * (1 - v.Progress) / v.Progress * float64(time.Second))
		}
		if best == nil || (eta > 0 && (best.ETA == 0 || eta > best.ETA)) {
			best = &BusyError{ETA: eta}
		}
	}
	if best == nil {
		return nil
	}
	return best
}

// NetworkBusy returns a *BusyError while a network task is running.
func (g *Gate) NetworkBusy() error {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.busyLocked()
}

// BeginLocal claims the GPU for a local generation. It fails with a *BusyError
// while a network task runs. Call the returned function when the run ends.
func (g *Gate) BeginLocal() (end func(), err error) {
	if g == nil {
		return func() {}, nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if err := g.busyLocked(); err != nil {
		return nil, err
	}
	g.local.Add(1)
	var once sync.Once
	return func() { once.Do(func() { g.local.Add(-1) }) }, nil
}

// Gate returns the gate shared by this process's worker slots.
func (a *Agent) Gate() *Gate { return a.gate }
