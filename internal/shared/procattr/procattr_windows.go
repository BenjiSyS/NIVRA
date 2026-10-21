//go:build windows

// Package procattr hides helper processes' console windows, which a
// console-less Nivra.exe would otherwise pop up for every child process.
package procattr

import (
	"os/exec"
	"syscall"
)

// CreateNoWindow is the Windows CREATE_NO_WINDOW process creation flag.
const CreateNoWindow = 0x08000000

// HideConsole starts cmd without a visible console window.
func HideConsole(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
	cmd.SysProcAttr.CreationFlags |= CreateNoWindow
}
