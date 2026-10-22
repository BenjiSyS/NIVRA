//go:build windows

package keepawake

import (
	"runtime"
	"syscall"
	"unsafe"
)

const (
	esContinuous     = 0x80000000
	esSystemRequired = 0x00000001
)

var (
	kernel32                 = syscall.NewLazyDLL("kernel32.dll")
	procSetThreadExecState   = kernel32.NewProc("SetThreadExecutionState")
	procGetSystemPowerStatus = kernel32.NewProc("GetSystemPowerStatus")
)

// winBackend owns one OS thread. SetThreadExecutionState with ES_CONTINUOUS
// applies to the calling thread, so the thread must stay the same until it is
// cleared; a goroutine locked to its thread does that.
type winBackend struct {
	req chan bool
}

func newBackend() backend {
	w := &winBackend{req: make(chan bool)}
	go w.loop()
	return w
}

func (w *winBackend) loop() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for on := range w.req {
		if on {
			procSetThreadExecState.Call(uintptr(esContinuous | esSystemRequired))
		} else {
			procSetThreadExecState.Call(uintptr(esContinuous))
		}
	}
}

func (w *winBackend) set(on bool) { w.req <- on }

type systemPowerStatus struct {
	ACLineStatus        byte
	BatteryFlag         byte
	BatteryLifePercent  byte
	SystemStatusFlag    byte
	BatteryLifeTime     uint32
	BatteryFullLifeTime uint32
}

func (w *winBackend) onBattery() bool {
	var st systemPowerStatus
	r, _, _ := procGetSystemPowerStatus.Call(uintptr(unsafe.Pointer(&st)))
	return r != 0 && st.ACLineStatus == 0 // 0 = offline (battery), 1 = AC, 255 = unknown
}
