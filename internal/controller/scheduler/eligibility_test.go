package scheduler

import (
	"strings"
	"testing"
)

func TestExplainWaiting(t *testing.T) {
	online := func(mod func(*deviceCaps)) deviceCaps {
		d := deviceCaps{online: true, executors: []string{"ai.refine_segment"}, ramMB: 8192, vramMB: 6144}
		if mod != nil {
			mod(&d)
		}
		return d
	}
	tests := []struct {
		name string
		devs []deviceCaps
		ram  int
		vram int
		want string // substring; "" means no reason
	}{
		{"no devices", nil, 1024, 0, "No computer has joined"},
		{"none online", []deviceCaps{{online: false}}, 1024, 0, "No helper computer is online"},
		{"all paused", []deviceCaps{online(func(d *deviceCaps) { d.paused = true })}, 1024, 0, "paused"},
		{"no executor", []deviceCaps{online(func(d *deviceCaps) { d.executors = nil })}, 1024, 0, "can make this kind of video"},
		{"vram too small", []deviceCaps{online(nil)}, 1024, 12288, "graphics memory"},
		{"ram too small", []deviceCaps{online(nil)}, 32768, 0, "enough memory"},
		{"all busy", []deviceCaps{online(func(d *deviceCaps) { d.busy = true })}, 1024, 4096, "busy"},
		{"someone can take it", []deviceCaps{online(nil)}, 1024, 4096, ""},
		{"rejected earlier by the only device", []deviceCaps{online(func(d *deviceCaps) { d.rejected = true })}, 1024, 0, "paused"},
	}
	for _, tc := range tests {
		got := explainWaiting("ai.refine_segment", tc.ram, tc.vram, tc.vram > 0, "", false, tc.devs)
		if tc.want == "" && got != "" || tc.want != "" && !strings.Contains(got, tc.want) {
			t.Errorf("%s: got %q, want containing %q", tc.name, got, tc.want)
		}
	}
	// A trusted-only job with no trusted online device.
	got := explainWaiting("ai.refine_segment", 0, 0, false, "", true, []deviceCaps{online(nil)})
	if !strings.Contains(got, "trusted") {
		t.Errorf("trusted: %q", got)
	}
}
