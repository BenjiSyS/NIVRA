package hardware

import (
	"context"
	"testing"
)

func TestProbeFFmpegMissingGivesPlainReason(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	p := ProbeFFmpeg(context.Background(), "")
	if p.OK || p.Error == "" {
		t.Fatalf("expected a missing-ffmpeg reason, got %+v", p)
	}
}

func TestProbeFFmpegBadConfiguredPathFallsBack(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	if p := ProbeFFmpeg(context.Background(), t.TempDir()); p.OK {
		t.Fatalf("empty folder must not count: %+v", p)
	}
}
