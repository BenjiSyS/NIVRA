package agent

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"time"

	"nivra/internal/shared/protocol"
	"nivra/internal/shared/sysinfo"
	"nivra/internal/shared/versioning"
	"nivra/internal/worker/client"
	"nivra/internal/worker/hardware"
)

var errRevoked = errors.New("device revoked")

func isRevoked(err error) bool {
	return protocol.IsCode(err, protocol.ErrRevoked) || protocol.IsCode(err, protocol.ErrNotEnrolled)
}

func jitter(d time.Duration) time.Duration {
	return d/2 + time.Duration(rand.Int64N(int64(d/2)+1))
}

func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

// loop keeps a session alive across network failures until ctx ends.
func (a *Agent) loop(ctx context.Context, done chan struct{}) {
	defer close(done)
	failures := 0
	for ctx.Err() == nil {
		a.setState(StateConnecting, "")
		a.mu.Lock()
		cl := a.cl
		a.mu.Unlock()
		if _, err := cl.OpenSession(ctx); err != nil {
			if ctx.Err() != nil {
				break
			}
			if isRevoked(err) {
				a.revoked(err)
				return
			}
			if errors.Is(err, client.ErrControllerIdentity) {
				a.setState(StateOffline, "controller identity mismatch; refusing to connect")
				a.logf("error", "session", "%v", err)
				if !sleepCtx(ctx, 5*time.Minute) {
					break
				}
				continue
			}
			failures++
			wait := jitter(min(time.Duration(1<<min(failures, 6))*time.Second, 60*time.Second))
			a.setState(StateOffline, fmt.Sprintf("controller unreachable (%v); retrying in %s", err, wait.Round(time.Second)))
			if !sleepCtx(ctx, wait) {
				break
			}
			continue
		}
		failures = 0
		a.logf("info", "session", "session established with %s", cl.Identity().ControllerURL)
		err := a.session(ctx, cl)
		if errors.Is(err, errRevoked) {
			return
		}
		if ctx.Err() != nil {
			break
		}
		a.logf("warn", "session", "session ended: %v; reconnecting", err)
		cl.InvalidateSession()
		if !sleepCtx(ctx, jitter(3*time.Second)) {
			break
		}
	}
	// The agent context is cancelled: wait for the running task to stop
	// (it reports STOPPED so the controller requeues it).
	if t := a.currentTask(); t != nil {
		select {
		case <-t.done:
		case <-time.After(90 * time.Second):
			a.logf("error", "jobs", "task did not stop within 90s")
		}
	}
	a.flushOutbox(context.Background(), 10*time.Second)
	a.shipLogs(context.Background())
}

func (a *Agent) revoked(err error) {
	a.setState(StateRevoked, "the project administrator revoked this device; connect with a new project code to continue")
	a.logf("error", "session", "%v", err)
	a.mu.Lock()
	a.running = false
	a.mu.Unlock()
	if t := a.currentTask(); t != nil {
		t.stop(stopDiscard)
	}
}

// session runs heartbeat and work loops until one fails.
func (a *Agent) session(ctx context.Context, cl *client.Client) error {
	sctx, cancel := context.WithCancel(ctx)
	defer cancel()
	errc := make(chan error, 2)
	// Reconcile results produced before a restart or disconnect.
	a.flushOutbox(sctx, 30*time.Second)
	go func() { errc <- a.heartbeatLoop(sctx, cl) }()
	go func() { errc <- a.workLoop(sctx, cl) }()
	err := <-errc
	cancel()
	// The work loop keeps the current task running across reconnects only
	// while ctx (the agent loop) lives; wait for the other goroutine.
	<-errc
	return err
}

func (a *Agent) currentTask() *activeTask {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.task
}

func (a *Agent) effectivelyPaused() (bool, string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	switch {
	case a.remotePaused:
		return true, "paused by the project administrator"
	case a.localPaused:
		return true, "paused on this computer"
	case !a.settings.InSchedule(time.Now()):
		return true, "outside scheduled operating hours"
	}
	return false, ""
}

func (a *Agent) heartbeatLoop(ctx context.Context, cl *client.Client) error {
	var lastRx, lastTx int64
	lastAt := time.Now()
	misses, beats := 0, 0
	var gpuPct float64
	var vramUsed int64
	var gpuAt time.Time
	for {
		a.mu.Lock()
		dirty := a.capsDirty
		gi := a.settings.GPUIndex
		a.mu.Unlock()
		req := protocol.HeartbeatRequest{State: protocol.WorkerIdle, AgentVersion: versioning.Version}
		if t := a.currentTask(); t != nil {
			req.State = protocol.WorkerBusy
			req.Attempts = append(req.Attempts, t.status())
		} else if p, _ := a.effectivelyPaused(); p {
			req.State = protocol.WorkerPaused
		}
		req.Attempts = append(req.Attempts, a.outboxAttempts()...)
		if dirty {
			c := a.capabilities()
			req.Capabilities = &c
		}
		// Metrics.
		mem, _ := sysinfo.ReadMemory()
		rx, tx := cl.Bytes()
		el := time.Since(lastAt).Seconds()
		if gi >= 0 && time.Since(gpuAt) > 10*time.Second {
			gpuPct, vramUsed = hardware.GPUUtilisation(ctx, gi)
			gpuAt = time.Now()
		}
		req.Metrics = protocol.MetricsSnapshot{CPUPct: sysinfo.CPUPercent(), RAMUsedMB: int64((mem.Total - mem.Available) >> 20),
			GPUPct: gpuPct, VRAMUsedMB: vramUsed}
		if el > 0 {
			req.Metrics.NetRxKbps = float64(rx-lastRx) * 8 / 1000 / el
			req.Metrics.NetTxKbps = float64(tx-lastTx) * 8 / 1000 / el
		}
		lastRx, lastTx, lastAt = rx, tx, time.Now()

		hctx, cancel := context.WithTimeout(ctx, 20*time.Second)
		resp, err := cl.Heartbeat(hctx, req)
		cancel()
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if isRevoked(err) {
				a.revoked(err)
				return errRevoked
			}
			misses++
			a.logf("warn", "heartbeat", "heartbeat failed (%d): %v", misses, err)
			if misses >= 3 {
				return fmt.Errorf("heartbeat failed %d times: %w", misses, err)
			}
		} else {
			misses = 0
			a.mu.Lock()
			if dirty {
				a.capsDirty = false
			}
			a.lastBeat = time.Now()
			wasPaused := a.remotePaused
			a.remotePaused = resp.Paused
			a.mu.Unlock()
			if wasPaused != resp.Paused {
				a.logf("info", "heartbeat", "controller pause flag is now %v", resp.Paused)
			}
			a.handleInvalid(resp.InvalidAttempts)
			if resp.PendingCommands > 0 {
				a.processCommands(ctx, cl)
			}
			a.refreshState()
			a.shipLogs(ctx)
			beats++
			a.mu.Lock()
			needSelf := a.self == nil || beats%4 == 0
			a.mu.Unlock()
			if needSelf {
				if me, err := cl.Me(ctx); err == nil {
					a.mu.Lock()
					a.self = me
					a.mu.Unlock()
				}
			}
		}
		if !sleepCtx(ctx, cl.HeartbeatInterval()) {
			return ctx.Err()
		}
	}
}

// refreshState derives the visible state from task/pause flags.
func (a *Agent) refreshState() {
	if t := a.currentTask(); t != nil {
		v := t.view()
		a.setState(StateBusy, v.JobType+" — "+v.Stage)
		return
	}
	if p, why := a.effectivelyPaused(); p {
		st := StatePaused
		if why == "outside scheduled operating hours" {
			st = StateOffHours
		}
		a.setState(st, why)
		return
	}
	a.setState(StateIdle, "ready for work")
}

// handleInvalid stops tasks the controller no longer considers ours.
func (a *Agent) handleInvalid(ids []string) {
	for _, id := range ids {
		if t := a.currentTask(); t != nil && t.a.AttemptID == id {
			a.logf("warn", "jobs", "controller reports attempt %s is no longer valid (lease expired or reassigned); discarding", id)
			t.stop(stopDiscard)
		}
		a.dropOutbox(id)
	}
}

func (a *Agent) processCommands(ctx context.Context, cl *client.Client) {
	cmds, err := cl.Commands(ctx)
	if err != nil {
		a.logf("warn", "commands", "fetch commands: %v", err)
		return
	}
	for _, c := range cmds {
		if c.State == protocol.CmdCompleted || c.State == protocol.CmdFailed {
			continue
		}
		a.logf("info", "commands", "received command %s (%s)", c.Type, c.ID)
		if c.State == protocol.CmdDelivered || c.State == protocol.CmdQueued {
			if err := cl.AckCommand(ctx, c.ID, protocol.CmdAcknowledged, nil); err != nil {
				a.logf("warn", "commands", "ack %s: %v", c.ID, err)
				continue
			}
		}
		switch c.Type {
		case protocol.CmdPause:
			a.mu.Lock()
			a.remotePaused = true
			a.mu.Unlock()
			_ = cl.AckCommand(ctx, c.ID, protocol.CmdCompleted, map[string]any{"paused": true})
		case protocol.CmdResume:
			a.mu.Lock()
			a.remotePaused = false
			a.mu.Unlock()
			_ = cl.AckCommand(ctx, c.ID, protocol.CmdCompleted, map[string]any{"paused": false})
		case protocol.CmdStopTask:
			_ = cl.AckCommand(ctx, c.ID, protocol.CmdExecuting, nil)
			t := a.currentTask()
			if t == nil {
				_ = cl.AckCommand(ctx, c.ID, protocol.CmdCompleted, map[string]any{"stopped": nil, "note": "no task was running"})
				continue
			}
			id := t.a.AttemptID
			t.stop(stopRequeue)
			go func(cmdID string) {
				// Confirm only once the task has actually stopped.
				select {
				case <-t.done:
					_ = cl.AckCommand(context.Background(), cmdID, protocol.CmdCompleted, map[string]any{"stopped": id})
				case <-time.After(2 * time.Minute):
					_ = cl.AckCommand(context.Background(), cmdID, protocol.CmdFailed, map[string]any{"error": "task did not stop within 2 minutes"})
				}
			}(c.ID)
		case protocol.CmdRefreshCapabilities:
			_ = cl.AckCommand(ctx, c.ID, protocol.CmdExecuting, nil)
			a.RefreshHardware(ctx)
			_ = cl.AckCommand(ctx, c.ID, protocol.CmdCompleted, nil)
		case protocol.CmdDisconnect:
			_ = cl.AckCommand(ctx, c.ID, protocol.CmdCompleted, map[string]any{"disconnecting": true})
			a.logf("warn", "commands", "controller requested disconnect")
			go a.Stop()
		default:
			_ = cl.AckCommand(ctx, c.ID, protocol.CmdFailed, map[string]any{"error": "unsupported command"})
		}
	}
}

func (a *Agent) shipLogs(ctx context.Context) {
	a.mu.Lock()
	cl := a.cl
	a.mu.Unlock()
	if cl == nil {
		return
	}
	lines, last := a.logs.unshipped(200)
	if len(lines) == 0 {
		a.logs.markShipped(last)
		return
	}
	sctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := cl.ShipLogs(sctx, lines); err == nil {
		a.logs.markShipped(last)
	}
}
