package agent

import (
	"testing"

	"nivra/internal/worker/executor"
	"nivra/internal/worker/hardware"
)

func TestNoFFmpegDropsVideoJobTypes(t *testing.T) {
	a := testAgent(t)
	a.hw.AI = hardware.AIProbe{OK: true, JobTypes: executor.PythonJobTypes}
	a.hw.FFmpeg = hardware.FFmpegProbe{Error: "ffmpeg is not installed (needed to save videos)"}
	a.rebuildExecutors()
	for _, v := range executor.VideoJobTypes {
		if _, ok := a.executors[v]; ok {
			t.Errorf("%s still advertised without ffmpeg", v)
		}
	}
	if _, ok := a.executors["ai.hd_train"]; !ok {
		t.Error("non-video job types must stay")
	}
	if st := a.Status(); st.VideoDisabledReason == "" {
		t.Error("no reason shown")
	}
	a.hw.FFmpeg = hardware.FFmpegProbe{OK: true}
	a.rebuildExecutors()
	if _, ok := a.executors["ai.generate"]; !ok {
		t.Error("video jobs missing with ffmpeg")
	}
	if st := a.Status(); st.VideoDisabledReason != "" {
		t.Error("reason should clear")
	}
}
