package keepawake

import "testing"

type fake struct {
	calls   []bool
	battery bool
}

func (f *fake) set(on bool)     { f.calls = append(f.calls, on) }
func (f *fake) onBattery() bool { return f.battery }

func TestRefCountedAcquire(t *testing.T) {
	f := &fake{}
	m := &Manager{b: f}
	r1 := m.Acquire(false)
	r2 := m.Acquire(false)
	if m.Held() != 2 || len(f.calls) != 1 || !f.calls[0] {
		t.Fatalf("held %d calls %v", m.Held(), f.calls)
	}
	r1()
	r1() // second call must be a no-op
	if m.Held() != 1 || len(f.calls) != 1 {
		t.Fatalf("after first release: held %d calls %v", m.Held(), f.calls)
	}
	r2()
	if m.Held() != 0 || len(f.calls) != 2 || f.calls[1] {
		t.Fatalf("after last release: held %d calls %v", m.Held(), f.calls)
	}
}

func TestAllowSleepOnBattery(t *testing.T) {
	f := &fake{battery: true}
	m := &Manager{b: f}
	rel := m.Acquire(true)
	rel()
	if len(f.calls) != 0 || m.Held() != 0 {
		t.Fatalf("battery + allow must not block sleep: %v", f.calls)
	}
	rel = m.Acquire(false) // setting off: always blocks
	if len(f.calls) != 1 {
		t.Fatalf("expected a block request, got %v", f.calls)
	}
	rel()
}

func TestDefaultManagerWorks(t *testing.T) {
	Acquire(false)()
}
