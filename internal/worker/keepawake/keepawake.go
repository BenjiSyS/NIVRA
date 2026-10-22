// Package keepawake stops the computer from going to sleep while Nivra works
// (a sleeping laptop kills an hour-long segment and burns an attempt). Only
// Windows is implemented; elsewhere it does nothing.
package keepawake

import "sync"

// backend switches the operating system's "stay awake" state.
type backend interface {
	// set(true) blocks sleep until set(false); it is called from one goroutine.
	set(on bool)
	// onBattery reports whether the computer currently runs on battery power.
	onBattery() bool
}

// Manager counts holders so several tasks share one OS request.
type Manager struct {
	b backend

	mu    sync.Mutex
	count int
}

// NewManager returns a Manager for the current operating system.
func NewManager() *Manager { return &Manager{b: newBackend()} }

// Default is the process-wide manager.
var Default = NewManager()

// Acquire blocks sleep until the returned release function is called (it is safe to
// call it more than once). When allowSleepOnBattery is true and the computer runs on
// battery, sleep is not blocked and release does nothing.
func (m *Manager) Acquire(allowSleepOnBattery bool) (release func()) {
	if allowSleepOnBattery && m.b.onBattery() {
		return func() {}
	}
	m.mu.Lock()
	m.count++
	if m.count == 1 {
		m.b.set(true)
	}
	m.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			m.mu.Lock()
			m.count--
			if m.count == 0 {
				m.b.set(false)
			}
			m.mu.Unlock()
		})
	}
}

// Held reports how many holders are active.
func (m *Manager) Held() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.count
}

// Acquire is Default.Acquire.
func Acquire(allowSleepOnBattery bool) func() { return Default.Acquire(allowSleepOnBattery) }
