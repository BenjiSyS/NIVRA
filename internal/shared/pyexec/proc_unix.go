//go:build !windows

package pyexec

import (
	"os/exec"
	"syscall"
)

// procGroup kills a process and its descendants via a POSIX process group.
type procGroup struct{ cmd *exec.Cmd }

func startGroup(cmd *exec.Cmd) (*procGroup, error) {
	cmd.SysProcAttr = sysProcAttr()
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &procGroup{cmd: cmd}, nil
}

// Kill sends SIGKILL to the whole process group.
func (g *procGroup) Kill() error {
	if g.cmd.Process == nil {
		return nil
	}
	return syscall.Kill(-g.cmd.Process.Pid, syscall.SIGKILL)
}

// Close kills any survivors of the group.
func (g *procGroup) Close() error {
	_ = g.Kill()
	return nil
}
