//go:build !linux && !windows

package sysinfo

import (
	"errors"
	"runtime"

	"golang.org/x/sys/unix"
)

// ReadMemory is unsupported on this platform.
func ReadMemory() (Memory, error) {
	return Memory{}, errors.New("memory info unsupported on " + runtime.GOOS)
}

func cpuTimes() (uint64, uint64, error) { return 0, 0, errors.New("unsupported") }

// CPUModel returns the architecture name.
func CPUModel() string { return runtime.GOARCH }

// PhysicalCores returns logical CPUs.
func PhysicalCores() int { return runtime.NumCPU() }

// DiskUsage uses statfs.
func DiskUsage(path string) (total, free uint64, err error) {
	var st unix.Statfs_t
	if err := unix.Statfs(path, &st); err != nil {
		return 0, 0, err
	}
	return st.Blocks * uint64(st.Bsize), st.Bavail * uint64(st.Bsize), nil
}

func uptime() float64 { return 0 }
