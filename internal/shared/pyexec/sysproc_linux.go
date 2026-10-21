//go:build linux

package pyexec

import "syscall"

// Pdeathsig kills the child if the worker dies.
func sysProcAttr() *syscall.SysProcAttr {
	return &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
}
