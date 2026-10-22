//go:build !windows

package keepawake

type noop struct{}

func (noop) set(bool)        {}
func (noop) onBattery() bool { return false }

func newBackend() backend { return noop{} }
