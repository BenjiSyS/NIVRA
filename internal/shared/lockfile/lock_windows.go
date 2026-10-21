//go:build windows

package lockfile

import (
	"os"

	"golang.org/x/sys/windows"
)

func lock(f *os.File) error {
	ol := new(windows.Overlapped)
	// Lock a byte range far beyond the PID text so reads of the file still work.
	ol.Offset = 0xFFFFFFF0
	return windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, ol)
}

func unlock(f *os.File) error {
	ol := new(windows.Overlapped)
	ol.Offset = 0xFFFFFFF0
	return windows.UnlockFileEx(windows.Handle(f.Fd()), 0, 1, 0, ol)
}
