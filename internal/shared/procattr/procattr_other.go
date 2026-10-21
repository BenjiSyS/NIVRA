//go:build !windows

// Package procattr hides helper processes' console windows on Windows.
package procattr

import "os/exec"

// HideConsole is a no-op outside Windows.
func HideConsole(*exec.Cmd) {}
