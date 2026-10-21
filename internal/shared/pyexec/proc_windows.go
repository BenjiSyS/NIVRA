//go:build windows

package pyexec

import (
	"os/exec"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// procGroup kills a process and all its descendants via a Windows job
// object. KILL_ON_JOB_CLOSE also kills them if the worker itself dies, so
// no orphaned training processes survive.
type procGroup struct {
	job windows.Handle
	cmd *exec.Cmd
}

func startGroup(cmd *exec.Cmd) (*procGroup, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, err
	}
	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err := windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info))); err != nil {
		windows.CloseHandle(job)
		return nil, err
	}
	// CREATE_NO_WINDOW: no console window pops up when Nivra.exe (no console) runs Python.
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: windows.CREATE_NEW_PROCESS_GROUP | 0x08000000, HideWindow: true}
	if err := cmd.Start(); err != nil {
		windows.CloseHandle(job)
		return nil, err
	}
	h, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(cmd.Process.Pid))
	if err == nil {
		err = windows.AssignProcessToJobObject(job, h)
		windows.CloseHandle(h)
	}
	if err != nil {
		_ = cmd.Process.Kill()
		windows.CloseHandle(job)
		return nil, err
	}
	return &procGroup{job: job, cmd: cmd}, nil
}

// Kill terminates every process in the group.
func (g *procGroup) Kill() error { return windows.TerminateJobObject(g.job, 1) }

// Close releases the job handle (killing any survivors).
func (g *procGroup) Close() error { return windows.CloseHandle(g.job) }
