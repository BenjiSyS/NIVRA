package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"nivra/internal/shared/config"
)

// Schedule restricts work to operating hours (local time).
type Schedule struct {
	Enabled bool   `json:"enabled"`
	Start   string `json:"start"` // HH:MM
	End     string `json:"end"`   // HH:MM; may wrap past midnight
}

// Settings are the user's local resource limits, persisted in settings.json.
type Settings struct {
	CPUThreads      int      `json:"cpu_threads"`
	GPUIndex        int      `json:"gpu_index"` // -1 = do not use a GPU
	RAMBudgetMB     int64    `json:"ram_budget_mb"`
	VRAMBudgetMB    int64    `json:"vram_budget_mb"`
	DiskCacheMB     int64    `json:"disk_cache_mb"`
	BandwidthKbps   int64    `json:"bandwidth_kbps"` // 0 = unlimited
	Schedule        Schedule `json:"schedule"`
	AutoStart       bool     `json:"auto_start"`
	ContributorName string   `json:"contributor_name"`
	// AllowSleepOnBattery: when true and the computer runs on battery, a running
	// task does not keep it awake. Default false: tasks always block sleep.
	AllowSleepOnBattery bool `json:"allow_sleep_on_battery"`
}

// DefaultSettings derives conservative defaults from the hardware.
func DefaultSettings(threads int, ramMB int64, hasGPU bool) Settings {
	s := Settings{CPUThreads: max(1, threads/2), GPUIndex: -1, RAMBudgetMB: ramMB / 2, DiskCacheMB: 20 * 1024,
		Schedule: Schedule{Start: "00:00", End: "23:59"}, AutoStart: true}
	if hasGPU {
		s.GPUIndex = 0
	}
	return s
}

func parseHM(s string) (int, error) {
	h, m, ok := strings.Cut(s, ":")
	if !ok {
		return 0, fmt.Errorf("time %q must be HH:MM", s)
	}
	hh, err1 := strconv.Atoi(h)
	mm, err2 := strconv.Atoi(m)
	if err1 != nil || err2 != nil || hh < 0 || hh > 23 || mm < 0 || mm > 59 {
		return 0, fmt.Errorf("time %q must be HH:MM", s)
	}
	return hh*60 + mm, nil
}

// Validate checks the settings against the detected hardware.
func (s Settings) Validate(threads int, ramMB int64, gpus int) error {
	var errs []error
	if s.CPUThreads < 1 || s.CPUThreads > threads {
		errs = append(errs, fmt.Errorf("cpu_threads must be 1..%d", threads))
	}
	if s.GPUIndex < -1 || s.GPUIndex >= gpus {
		errs = append(errs, fmt.Errorf("gpu_index must be -1 (none) or 0..%d", gpus-1))
	}
	if s.RAMBudgetMB < 256 || s.RAMBudgetMB > ramMB {
		errs = append(errs, fmt.Errorf("ram_budget_mb must be 256..%d", ramMB))
	}
	if s.VRAMBudgetMB < 0 || s.DiskCacheMB < 512 || s.BandwidthKbps < 0 {
		errs = append(errs, errors.New("vram_budget_mb >= 0, disk_cache_mb >= 512 and bandwidth_kbps >= 0 required"))
	}
	if _, err := parseHM(s.Schedule.Start); err != nil {
		errs = append(errs, err)
	}
	if _, err := parseHM(s.Schedule.End); err != nil {
		errs = append(errs, err)
	}
	if len(s.ContributorName) > 64 {
		errs = append(errs, errors.New("contributor_name too long"))
	}
	return errors.Join(errs...)
}

// InSchedule reports whether t falls inside operating hours.
func (s Settings) InSchedule(t time.Time) bool {
	if !s.Schedule.Enabled {
		return true
	}
	start, err1 := parseHM(s.Schedule.Start)
	end, err2 := parseHM(s.Schedule.End)
	if err1 != nil || err2 != nil {
		return true
	}
	now := t.Hour()*60 + t.Minute()
	if start <= end {
		return now >= start && now < end
	}
	return now >= start || now < end // wraps midnight
}

func loadSettings(dir string) (*Settings, error) {
	b, err := os.ReadFile(filepath.Join(dir, "settings.json"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var s Settings
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, fmt.Errorf("settings.json corrupt: %w", err)
	}
	return &s, nil
}

func saveSettings(dir string, s Settings) error {
	return config.Save(filepath.Join(dir, "settings.json"), s)
}
