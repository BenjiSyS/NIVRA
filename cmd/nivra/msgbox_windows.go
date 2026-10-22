//go:build windows

package main

import (
	"syscall"
	"unsafe"
)

var procMessageBoxW = syscall.NewLazyDLL("user32.dll").NewProc("MessageBoxW")

// showMessageBox shows a native error box. Nivra.exe has no console, so this is
// the only way a start-up failure is visible when the exe is double-clicked.
func showMessageBox(title, text string) {
	t, err1 := syscall.UTF16PtrFromString(text)
	c, err2 := syscall.UTF16PtrFromString(title)
	if err1 != nil || err2 != nil {
		return
	}
	const mbOK, mbIconError, mbSetForeground = 0x0, 0x10, 0x10000
	_, _, _ = procMessageBoxW.Call(0, uintptr(unsafe.Pointer(t)), uintptr(unsafe.Pointer(c)), uintptr(mbOK|mbIconError|mbSetForeground))
}
