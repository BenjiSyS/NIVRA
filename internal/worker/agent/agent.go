// Package agent is the Nivra Worker: it enrolls with a project, keeps a
// signed session alive, reports capabilities and resources, executes leased
// jobs and obeys reliable controller commands.
package agent

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"nivra/internal/shared/config"
	"nivra/internal/shared/protocol"
	"nivra/internal/shared/secure"
	"nivra/internal/shared/versioning"
	"nivra/internal/worker/client"
	"nivra/internal/worker/executor"
	"nivra/internal/worker/hardware"
	"nivra/internal/worker/identity"
)

// Agent states shown in the Worker website.
const (
	StateNotEnrolled = "not_enrolled"
	StateStopped     = "stopped"
	StateConnecting  = "connecting"
	StateIdle        = "idle"
	StateBusy        = "busy"
	StatePaused      = "paused"
	StateOffHours    = "outside_schedule"
	StateOffline     = "offline" // controller unreachable; retrying
	StateRevoked     = "revoked"
)

// Agent is the worker runtime. All exported methods are safe for concurrent use.
type Agent struct {
	cfg  config.Worker
	ids  *identity.Store
	logs *logRing
	gate *Gate
	// diskFree overrides sysinfo.DiskFree (tests).
	diskFree func(string) (uint64, error)

	mu           sync.Mutex
	id           *identity.Identity
	cl           *client.Client
	settings     Settings
	hw           hardware.Report
	running      bool
	localPaused  bool
	remotePaused bool
	state        string
	detail       string
	task         *activeTask
	self         *protocol.WorkerSelf
	lastBeat     time.Time
	capsDirty    bool
	executors    map[string]executor.Executor
	loopCancel   context.CancelFunc
	loopCtx      context.Context
	loopDone     chan struct{}
	completed    int
	slot         int // GPU slot (0 = primary worker)
	failed       int
}

// New loads identity and settings and detects hardware.
func New(ctx context.Context, cfg config.Worker) (*Agent, error) {
	for _, d := range []string{"", "work", "cache", "outbox", "logs"} {
		if err := os.MkdirAll(filepath.Join(cfg.DataDir, d), 0o700); err != nil {
			return nil, err
		}
	}
	a := &Agent{cfg: cfg, ids: identity.NewStore(cfg.DataDir), logs: newLogRing(2000), state: StateNotEnrolled, gate: NewGate()}
	a.gate.register(a)
	id, err := a.ids.Load()
	if err != nil {
		return nil, fmt.Errorf("load identity: %w", err)
	}
	a.id = id
	a.detectHardware(ctx)
	s, err := loadSettings(cfg.DataDir)
	if err != nil {
		return nil, err
	}
	ram := int64(a.hw.Host.MemTotal >> 20)
	if s == nil {
		d := DefaultSettings(a.hw.Host.Threads, ram, len(a.hw.GPUs) > 0 && a.hw.AI.CUDA)
		s = &d
		if err := saveSettings(cfg.DataDir, d); err != nil {
			return nil, err
		}
	}
	a.settings = *s
	if a.id != nil {
		a.state = StateStopped
	}
	a.rebuildExecutors()
	a.logf("info", "agent", "Nivra Worker %s on %s; %s", versioning.Version, versioning.Platform(), a.hwSummary())
	return a, nil
}

func (a *Agent) logf(level, service, format string, args ...any) {
	a.logs.add(level, service, fmt.Sprintf(format, args...))
}

func (a *Agent) hwSummary() string {
	g := "no NVIDIA GPU"
	if len(a.hw.GPUs) > 0 {
		g = fmt.Sprintf("%s (%d MiB)", a.hw.GPUs[0].Name, a.hw.GPUs[0].VRAMMB)
	}
	ai := "AI executor unavailable"
	if a.hw.AI.OK {
		ai = fmt.Sprintf("AI executor ready (torch %s, cuda=%v)", a.hw.AI.Torch, a.hw.AI.CUDA)
	} else if a.hw.AI.Error != "" {
		ai += " (" + a.hw.AI.Error + ")"
	}
	return fmt.Sprintf("%s, %d threads, %d MiB RAM, %s, %s", a.hw.Host.CPUModel, a.hw.Host.Threads, a.hw.Host.MemTotal>>20, g, ai)
}

func (a *Agent) detectHardware(ctx context.Context) {
	hw := hardware.Detect(ctx, a.cfg.DataDir, a.cfg.PythonPath, a.cfg.AIPackageDir, a.cfg.FFmpegPath)
	a.mu.Lock()
	a.hw = hw
	a.capsDirty = true
	a.mu.Unlock()
}

// rebuildExecutors must be called with settings/hardware current.
func (a *Agent) rebuildExecutors() {
	ex := map[string]executor.Executor{}
	b := executor.Builtin{}
	for _, t := range b.Types() {
		ex[t] = b
	}
	if a.hw.AI.OK && len(a.hw.AI.JobTypes) > 0 {
		p := &executor.Python{Interpreter: a.cfg.PythonPath, AIDir: a.cfg.AIPackageDir, FFmpeg: a.cfg.FFmpegPath,
			JobTypes: a.hw.AI.JobTypes}
		for _, t := range p.Types() {
			// Without ffmpeg a video job would render for minutes and then fail at
			// encoding, so the controller must not hand such jobs to this worker.
			if !a.hw.FFmpeg.OK && executor.IsVideoJobType(t) {
				continue
			}
			ex[t] = p
		}
	}
	a.executors = ex
}

// capabilities builds the capability report from hardware and limits.
func (a *Agent) capabilities() protocol.Capabilities {
	a.mu.Lock()
	defer a.mu.Unlock()
	c := protocol.Capabilities{SchemaVersion: protocol.CapabilitiesSchemaVersion, OS: runtime.GOOS, Arch: runtime.GOARCH,
		CPU:  protocol.CPUInfo{Model: a.hw.Host.CPUModel, Cores: a.hw.Host.Cores, Threads: a.hw.Host.Threads},
		GPUs: a.hw.GPUs, RAMMB: int64(a.hw.Host.MemTotal >> 20), RAMBudgetMB: a.settings.RAMBudgetMB,
		StorageFreeMB: int64(a.hw.Host.DiskFree >> 20), Backends: []string{"cpu"}}
	if c.GPUs == nil {
		c.GPUs = []protocol.GPUInfo{}
	}
	if gi := a.settings.GPUIndex; gi >= 0 && gi < len(a.hw.GPUs) && a.hw.AI.CUDA {
		c.VRAMMB = a.hw.GPUs[gi].VRAMMB
		c.VRAMBudgetMB = a.settings.VRAMBudgetMB
		if c.VRAMBudgetMB <= 0 || c.VRAMBudgetMB > c.VRAMMB {
			c.VRAMBudgetMB = c.VRAMMB * 9 / 10
		}
		c.Backends = append(c.Backends, "cuda")
		c.GPUs[gi].CUDAAvailable = true
	}
	for t := range a.executors {
		c.Executors = append(c.Executors, t)
	}
	return c
}

// Status is the Worker website's view of the agent.
type Status struct {
	State          string                     `json:"state"`
	Detail         string                     `json:"detail"`
	Enrolled       bool                       `json:"enrolled"`
	Running        bool                       `json:"running"`
	LocalPaused    bool                       `json:"local_paused"`
	RemotePaused   bool                       `json:"remote_paused"`
	ControllerURL  string                     `json:"controller_url,omitempty"`
	ProjectID      string                     `json:"project_id,omitempty"`
	ProjectName    string                     `json:"project_name,omitempty"`
	DeviceID       string                     `json:"device_id,omitempty"`
	DeviceName     string                     `json:"device_name,omitempty"`
	ControllerKey  string                     `json:"controller_fingerprint,omitempty"`
	LastHeartbeat  *time.Time                 `json:"last_heartbeat,omitempty"`
	Task           *TaskView                  `json:"task"`
	Hardware       hardware.Report            `json:"hardware"`
	Settings       Settings                   `json:"settings"`
	Executors      []string                   `json:"executors"`
	Contributions  []protocol.ContributionSum `json:"contributions"`
	CompletedLocal int                        `json:"completed_this_session"`
	FailedLocal    int                        `json:"failed_this_session"`
	Version        string                     `json:"version"`
	// VideoDisabledReason is set when video jobs are switched off on this computer.
	VideoDisabledReason string `json:"video_disabled_reason,omitempty"`
}

// Status returns a snapshot.
func (a *Agent) Status() Status {
	a.mu.Lock()
	defer a.mu.Unlock()
	st := Status{State: a.state, Detail: a.detail, Enrolled: a.id != nil, Running: a.running, LocalPaused: a.localPaused,
		RemotePaused: a.remotePaused, Hardware: a.hw, Settings: a.settings, CompletedLocal: a.completed, FailedLocal: a.failed,
		Version: versioning.Version, Contributions: []protocol.ContributionSum{}}
	if a.id != nil {
		st.ControllerURL, st.ProjectID, st.ProjectName = a.id.ControllerURL, a.id.ProjectID, a.id.ProjectName
		st.DeviceID, st.DeviceName = a.id.DeviceID, a.id.DeviceName
		st.ControllerKey = secure.Fingerprint(a.id.ControllerKey)
	}
	if !a.lastBeat.IsZero() {
		t := a.lastBeat
		st.LastHeartbeat = &t
	}
	if a.hw.AI.OK && !a.hw.FFmpeg.OK {
		st.VideoDisabledReason = "Video jobs are off on this computer: " + a.hw.FFmpeg.Error +
			". Install ffmpeg (Nivra-Setup.bat does this) or set ffmpeg_path in the worker settings, then press Refresh."
	}
	if a.task != nil {
		v := a.task.view()
		st.Task = &v
	}
	if a.self != nil {
		st.Contributions = a.self.Contributions
	}
	for t := range a.executors {
		st.Executors = append(st.Executors, t)
	}
	return st
}

// Logs returns recent local log lines after seq.
func (a *Agent) Logs(after int64, limit int) []LogLine { return a.logs.since(after, limit) }

// setState updates the visible state.
func (a *Agent) setState(state, detail string) {
	a.mu.Lock()
	changed := a.state != state || a.detail != detail
	a.state, a.detail = state, detail
	a.mu.Unlock()
	if changed {
		msg := state
		if detail != "" {
			msg += ": " + detail
		}
		a.logf("info", "agent", "state %s", msg)
	}
}

// Connect enrolls with a project code and starts working.
func (a *Agent) Connect(ctx context.Context, controllerURL, code, deviceName, contributor string) error {
	controllerURL = strings.TrimSpace(controllerURL)
	if controllerURL == "" {
		controllerURL = a.cfg.ControllerURL
	}
	if controllerURL == "" {
		return errors.New("controller URL required")
	}
	if strings.TrimSpace(deviceName) == "" {
		h, _ := os.Hostname()
		deviceName = h
	}
	a.mu.Lock()
	if contributor == "" {
		contributor = a.settings.ContributorName
	}
	old := a.id
	a.mu.Unlock()
	var pinned []byte
	if old != nil && strings.EqualFold(strings.TrimRight(old.ControllerURL, "/"), strings.TrimRight(controllerURL, "/")) {
		pinned = old.ControllerKey // same controller: require the same identity
	}
	a.logf("info", "enrollment", "enrolling with %s", controllerURL)
	res, err := client.Enroll(ctx, controllerURL, a.cfg.AllowInsecureHTTP, code,
		protocol.DeviceInfo{Name: deviceName, ContributorName: contributor, OS: runtime.GOOS, Arch: runtime.GOARCH}, pinned)
	if err != nil {
		a.logf("error", "enrollment", "enrollment failed: %v", err)
		return err
	}
	// Leaving the old project: stop work there first (tasks are requeued).
	a.Stop()
	if err := a.ids.Save(res.Identity); err != nil {
		return err
	}
	a.mu.Lock()
	a.id = res.Identity
	a.self = nil
	a.remotePaused = false
	a.mu.Unlock()
	a.logf("info", "enrollment", "enrolled as %q in project %q (controller key %s)", res.Identity.DeviceName, res.Identity.ProjectName, res.ControllerFingerprint)
	return a.Start()
}

// Start begins the connection/work loop.
func (a *Agent) Start() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.id == nil {
		return errors.New("not enrolled: connect with a project code first")
	}
	if a.running {
		return nil
	}
	cl, err := client.New(a.id, a.cfg.AllowInsecureHTTP)
	if err != nil {
		return err
	}
	cl.Limit.Store(a.settings.BandwidthKbps * 1000 / 8)
	a.cl = cl
	ctx, cancel := context.WithCancel(context.Background())
	a.loopCancel, a.loopCtx, a.loopDone, a.running = cancel, ctx, make(chan struct{}), true
	a.capsDirty = true
	go a.loop(ctx, a.loopDone)
	return nil
}

func (a *Agent) loopCtxLocked() context.Context { return a.loopCtx }

// Stop gracefully stops the current task (it is requeued by the controller)
// and disconnects. It blocks until the loop exits.
func (a *Agent) Stop() {
	a.mu.Lock()
	cancel, done := a.loopCancel, a.loopDone
	a.loopCancel, a.loopDone = nil, nil
	a.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	<-done
	a.mu.Lock()
	a.running = false
	a.mu.Unlock()
	if a.id != nil {
		a.setState(StateStopped, "")
	}
}

// Pause stops taking new jobs; the current job finishes.
func (a *Agent) Pause() {
	a.mu.Lock()
	a.localPaused = true
	a.mu.Unlock()
	a.logf("info", "agent", "paused locally: no new jobs will be accepted")
}

// Resume accepts jobs again.
func (a *Agent) Resume() {
	a.mu.Lock()
	a.localPaused = false
	a.mu.Unlock()
	a.logf("info", "agent", "resumed locally")
}

// StopTask gracefully stops the current task only.
func (a *Agent) StopTask() bool {
	a.mu.Lock()
	t := a.task
	a.mu.Unlock()
	if t == nil {
		return false
	}
	t.stop(stopRequeue)
	return true
}

// UpdateSettings validates, persists and applies new limits.
func (a *Agent) UpdateSettings(s Settings) error {
	a.mu.Lock()
	threads, ram, gpus := a.hw.Host.Threads, int64(a.hw.Host.MemTotal>>20), len(a.hw.GPUs)
	a.mu.Unlock()
	if err := s.Validate(threads, ram, gpus); err != nil {
		return err
	}
	if err := saveSettings(a.cfg.DataDir, s); err != nil {
		return err
	}
	a.mu.Lock()
	a.settings = s
	a.capsDirty = true
	if a.cl != nil {
		a.cl.Limit.Store(s.BandwidthKbps * 1000 / 8)
	}
	a.mu.Unlock()
	a.logf("info", "agent", "settings updated")
	return nil
}

// RefreshHardware re-detects hardware and re-reports capabilities.
func (a *Agent) RefreshHardware(ctx context.Context) {
	a.detectHardware(ctx)
	a.mu.Lock()
	a.rebuildExecutors()
	a.mu.Unlock()
	a.logf("info", "agent", "hardware refreshed: %s", a.hwSummary())
}

// AutoStart starts the loop if enrolled and configured to do so.
func (a *Agent) AutoStart() {
	a.mu.Lock()
	ok := a.id != nil && a.settings.AutoStart
	a.mu.Unlock()
	if ok {
		if err := a.Start(); err != nil {
			a.logf("error", "agent", "auto start failed: %v", err)
		}
	}
}

// Shutdown stops everything (used on process exit).
func (a *Agent) Shutdown() { a.Stop() }
