package training

import (
	"errors"
	"testing"

	"nivra/internal/controller/core"
)

// Start on the Small slot creates one run from the newest small version; a
// second Start returns the same run; Stop pauses it and Start resumes it. The
// HD slots refuse to start without a starting model.
func TestSlotsStartStop(t *testing.T) {
	e := newHDEnv(t)
	w := e.art(e.project, "model_weights", randBytes(64))
	m := e.art(e.project, "model_version", randBytes(64))
	vae := e.art(e.project, "vae_weights", randBytes(64)).ID
	vid := e.version(e.project, map[string]any{"config_name": "nivra_small"}, w, m, &vae)

	views, err := e.s.Slots(e.ctx, e.project)
	if err != nil {
		t.Fatal(err)
	}
	if len(views) != 4 || views[0].Key != "small" || views[0].Blocked != "" || views[0].VersionID == nil || *views[0].VersionID != vid {
		t.Fatalf("small slot view: %+v", views[0])
	}
	if views[1].Blocked == "" {
		t.Fatal("medium slot should be blocked without a starting model")
	}

	run, err := e.s.StartSlot(e.ctx, core.System, e.project, "small")
	if err != nil {
		t.Fatal(err)
	}
	if run.Name != "Small" || run.Status != "created" || run.Config.InitWeightsArtifact != w.ID || run.Config.Rounds != slotRounds {
		t.Fatalf("run: %+v", run)
	}
	again, err := e.s.StartSlot(e.ctx, core.System, e.project, "small")
	if err != nil || again.ID != run.ID {
		t.Fatalf("second start: %v %+v", err, again)
	}
	p, err := e.s.StopSlot(e.ctx, core.System, e.project, "small")
	if err != nil || p.Status != "paused" {
		t.Fatalf("stop: %v %+v", err, p)
	}
	r, err := e.s.StartSlot(e.ctx, core.System, e.project, "small")
	if err != nil || r.ID != run.ID || r.Status != "running" {
		t.Fatalf("resume: %v %+v", err, r)
	}

	if _, err := e.s.StartSlot(e.ctx, core.System, e.project, "medium"); !errors.Is(err, ErrBadConfig) {
		t.Fatalf("medium start: %v", err)
	}
	if _, err := e.s.StartSlot(e.ctx, core.System, e.project, "nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unknown slot: %v", err)
	}
}
