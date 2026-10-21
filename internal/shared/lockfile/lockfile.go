// Package lockfile provides an exclusive, OS-released instance lock so two
// controllers or two workers cannot run on the same data directory.
package lockfile

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ErrLocked means another process holds the lock.
var ErrLocked = errors.New("another Nivra process is already using this data directory")

// Lock is a held instance lock.
type Lock struct {
	f    *os.File
	path string
}

// Acquire takes an exclusive non-blocking lock on path. The OS releases it
// automatically if the process dies, so stale locks never block startup.
func Acquire(path string) (*Lock, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	if err := lock(f); err != nil {
		holder, _ := os.ReadFile(path)
		f.Close()
		if pid := strings.TrimSpace(string(holder)); pid != "" {
			return nil, fmt.Errorf("%w (held by pid %s, lock file %s)", ErrLocked, pid, path)
		}
		return nil, fmt.Errorf("%w (lock file %s)", ErrLocked, path)
	}
	_ = f.Truncate(0)
	_, _ = f.WriteAt([]byte(strconv.Itoa(os.Getpid())+"\n"), 0)
	return &Lock{f: f, path: path}, nil
}

// Release unlocks and closes the lock file.
func (l *Lock) Release() error {
	if l == nil || l.f == nil {
		return nil
	}
	_ = l.f.Truncate(0)
	_ = unlock(l.f)
	err := l.f.Close()
	l.f = nil
	return err
}
