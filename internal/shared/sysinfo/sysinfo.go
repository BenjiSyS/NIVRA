// Package sysinfo reads host resources (CPU, memory, disk) without cgo.
package sysinfo

import (
	"runtime"
	"sync"
	"time"
)

// Memory describes physical memory in bytes.
type Memory struct {
	Total     uint64 `json:"total"`
	Available uint64 `json:"available"`
}

// Host is a snapshot of host information.
type Host struct {
	OS        string  `json:"os"`
	Arch      string  `json:"arch"`
	CPUModel  string  `json:"cpu_model"`
	Cores     int     `json:"cores"`
	Threads   int     `json:"threads"`
	MemTotal  uint64  `json:"mem_total"`
	MemAvail  uint64  `json:"mem_available"`
	CPUPct    float64 `json:"cpu_pct"`
	DiskTotal uint64  `json:"disk_total"`
	DiskFree  uint64  `json:"disk_free"`
	Uptime    float64 `json:"uptime_sec,omitempty"`
}

// CPUSampler computes CPU utilisation between successive calls.
type CPUSampler struct {
	mu         sync.Mutex
	idle, tot  uint64
	last       float64
	lastSample time.Time
}

// Percent returns busy CPU % since the previous call (0 on the first call).
// Calls closer than 500 ms apart return the cached value.
func (s *CPUSampler) Percent() float64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	if time.Since(s.lastSample) < 500*time.Millisecond {
		return s.last
	}
	idle, tot, err := cpuTimes()
	if err != nil {
		return 0
	}
	if s.tot != 0 && tot > s.tot {
		di, dt := float64(idle-s.idle), float64(tot-s.tot)
		s.last = 100 * (1 - di/dt)
		if s.last < 0 {
			s.last = 0
		}
	}
	s.idle, s.tot, s.lastSample = idle, tot, time.Now()
	return s.last
}

var defaultSampler CPUSampler

// CPUPercent returns process-wide sampled CPU utilisation.
func CPUPercent() float64 { return defaultSampler.Percent() }

// Snapshot collects host information; path selects the disk to report.
func Snapshot(path string) Host {
	h := Host{OS: runtime.GOOS, Arch: runtime.GOARCH, CPUModel: CPUModel(), Threads: runtime.NumCPU(), Cores: PhysicalCores()}
	if m, err := ReadMemory(); err == nil {
		h.MemTotal, h.MemAvail = m.Total, m.Available
	}
	h.CPUPct = defaultSampler.Percent()
	if t, f, err := DiskUsage(path); err == nil {
		h.DiskTotal, h.DiskFree = t, f
	}
	h.Uptime = uptime()
	return h
}

// DiskFree returns free bytes available to the current user at path.
func DiskFree(path string) (uint64, error) {
	_, f, err := DiskUsage(path)
	return f, err
}
