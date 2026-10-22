package training

import (
	"errors"
	"strings"
	"testing"

	"nivra/internal/controller/api"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/controller/projects"
)

func (e *hdEnv) member(name, role string) string {
	e.t.Helper()
	var id string
	if err := e.s.d.DB.QueryRow(e.ctx, `INSERT INTO users (username, password_hash, role) VALUES ($1,'x',$2) RETURNING id::text`, name, role).Scan(&id); err != nil {
		e.t.Fatal(err)
	}
	return id
}

func (e *hdEnv) genStatus(gid string) (status, stage, reason string, attention bool) {
	e.t.Helper()
	if err := e.s.d.DB.QueryRow(e.ctx, `SELECT status, stage, failure_reason, needs_attention_since IS NOT NULL FROM generations WHERE id=$1`, gid).
		Scan(&status, &stage, &reason, &attention); err != nil {
		e.t.Fatal(err)
	}
	return
}

func (e *hdEnv) advance(gid string) {
	e.t.Helper()
	if err := e.s.advanceGeneration(e.ctx, gid); err != nil {
		e.t.Fatal(err)
	}
}

func (e *hdEnv) childIDs(gid, stage string) []string {
	e.t.Helper()
	rows, err := e.s.d.DB.Query(e.ctx, `SELECT job_id::text FROM generation_jobs WHERE generation_id=$1 AND stage=$2 ORDER BY idx`, gid, stage)
	if err != nil {
		e.t.Fatal(err)
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			e.t.Fatal(err)
		}
		ids = append(ids, id)
	}
	return ids
}

func (e *hdEnv) artState(id string) string {
	e.t.Helper()
	var st string
	if err := e.s.d.DB.QueryRow(e.ctx, `SELECT state FROM artifacts WHERE id=$1`, id).Scan(&st); err != nil {
		e.t.Fatal(err)
	}
	return st
}

// A long video: a failed part keeps the video running ("needs attention"), the
// other parts keep going, the owner can regenerate the part, and the finished
// video loses its draft and segment files but keeps the final video.
func TestGenerationFailedPartNeedsAttentionAndRegenerate(t *testing.T) {
	e := newHDEnv(t)
	uid := e.member("owner-one", "member")
	dev := e.device(e.project, "d1")
	gp := genParams{Prompt: "a cat", Seed: 1, Width: 1280, Height: 720, FPS: 30, Seconds: 20, SegFrame: segmentFrames(8, 30), Frames: 601,
		Inputs: []string{newUUID(), newUUID(), newUUID()}, RAMMB: 1024}
	gid, draft, err := e.s.createGeneration(e.ctx, e.project, newUUID(), &uid, gp)
	if err != nil {
		t.Fatal(err)
	}
	e.completeJob(draft, dev, "draft_video", randBytes(300), nil)
	e.advance(gid)
	refines := e.childIDs(gid, "refine")
	if len(refines) != 3 {
		t.Fatalf("refine jobs: %d", len(refines))
	}
	// Videos share the interactive priority band and later stages go first.
	var prio, attempts int
	if err := e.s.d.DB.QueryRow(e.ctx, `SELECT priority, max_attempts FROM jobs WHERE id=$1`, refines[0]).Scan(&prio, &attempts); err != nil {
		t.Fatal(err)
	}
	if prio/10 != 2 || attempts != 6 {
		t.Fatalf("refine priority %d attempts %d, want band 2 and 6 attempts", prio, attempts)
	}

	// One part runs out of attempts.
	if _, err := e.s.d.DB.Exec(e.ctx, `UPDATE jobs SET status='failed', failure_reason='INTERNAL: ffmpeg not found' WHERE id=$1`, refines[1]); err != nil {
		t.Fatal(err)
	}
	e.advance(gid)
	status, stage, reason, attention := e.genStatus(gid)
	if status != "running" || stage != "refine" || !attention || !strings.Contains(reason, "refine part 1") || !strings.Contains(reason, "ffmpeg") {
		t.Fatalf("after a failed part: %s/%s attention=%v reason=%q", status, stage, attention, reason)
	}
	var cancelled int
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT count(*) FROM jobs j JOIN generation_jobs g ON g.job_id=j.id WHERE g.generation_id=$1 AND j.status='cancelled'`, gid).Scan(&cancelled)
	if cancelled != 0 {
		t.Fatalf("siblings of the failed part must keep running, %d were cancelled", cancelled)
	}
	v, err := e.s.generationView(e.ctx, e.project, gid, uid, true)
	if err != nil || v["needs_attention"] != true {
		t.Fatalf("view: %v %v", v, err)
	}
	if _, err := e.s.generationView(e.ctx, e.project, gid, e.member("someone-else", "member"), false); !errors.Is(err, errGenNotFound) {
		t.Fatalf("another member must not see the video: %v", err)
	}

	n, err := e.s.RegenerateParts(e.ctx, core.System, e.project, gid, "", nil)
	if err != nil || n != 1 {
		t.Fatalf("regenerate: %d %v", n, err)
	}
	if _, err := e.s.RegenerateParts(e.ctx, core.System, e.project, gid, "", nil); !errors.Is(err, ErrBadState) {
		t.Fatalf("nothing left to regenerate: %v", err)
	}
	var st string
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT status FROM jobs WHERE id=$1`, refines[1]).Scan(&st)
	if st != "ready" {
		t.Fatalf("revived job is %s", st)
	}
	e.advance(gid)
	if _, _, _, attention := e.genStatus(gid); attention {
		t.Fatal("attention flag must clear once the part is back in the queue")
	}

	var segArts []string
	for _, id := range refines {
		att := e.completeJob(id, dev, "segment_video", randBytes(200), nil)
		var a string
		_ = e.s.d.DB.QueryRow(e.ctx, `SELECT id::text FROM artifacts WHERE attempt_id=$1`, att).Scan(&a)
		segArts = append(segArts, a)
	}
	e.advance(gid)
	if _, stage, _, _ := e.genStatus(gid); stage != "stitch" {
		t.Fatalf("stage %s, want stitch", stage)
	}
	stitchAtt := e.completeJob(e.childIDs(gid, "stitch")[0], dev, "video", randBytes(400), nil)
	e.advance(gid)
	if status, _, _, _ := e.genStatus(gid); status != "completed" {
		t.Fatalf("status %s", status)
	}
	var final string
	_ = e.s.d.DB.QueryRow(e.ctx, `SELECT id::text FROM artifacts WHERE attempt_id=$1`, stitchAtt).Scan(&final)

	// Retention: nothing is old enough, then a forced cleanup removes only intermediates.
	if n, _, err := e.s.PurgeIntermediates(e.ctx, e.project, false); err != nil || n != 0 {
		t.Fatalf("retention purge removed %d (%v) files of a video that just finished", n, err)
	}
	if _, err := e.s.d.DB.Exec(e.ctx, `UPDATE generations SET completed_at = now() - interval '25 hours' WHERE id=$1`, gid); err != nil {
		t.Fatal(err)
	}
	n2, freed, err := e.s.PurgeIntermediates(e.ctx, e.project, false)
	if err != nil || n2 != 4 || freed == 0 {
		t.Fatalf("purge after 25 h: %d files, %d bytes, %v (draft + 3 segments expected)", n2, freed, err)
	}
	for _, a := range segArts {
		if e.artState(a) != "deleted" {
			t.Fatal("segment must be deleted")
		}
	}
	if e.artState(final) != "complete" {
		t.Fatal("the final video must be kept")
	}

	// Deleting the video removes its final file too.
	if _, err := e.s.DeleteGeneration(e.ctx, core.System, e.project, gid); err != nil {
		t.Fatal(err)
	}
	if e.artState(final) != "deleted" {
		t.Fatal("the final video must be deleted with the generation")
	}
	if _, err := e.s.DeleteGeneration(e.ctx, core.System, e.project, gid); !errors.Is(err, ErrNotFound) {
		t.Fatalf("second delete: %v", err)
	}
}

func TestDeleteRunningGenerationIsRefused(t *testing.T) {
	e := newHDEnv(t)
	uid := e.member("owner-two", "member")
	gp := genParams{Prompt: "x", Seed: 1, Width: 1280, Height: 720, FPS: 30, Seconds: 20, SegFrame: segmentFrames(8, 30), Frames: 601,
		Inputs: []string{newUUID(), newUUID(), newUUID()}}
	gid, _, err := e.s.createGeneration(e.ctx, e.project, newUUID(), &uid, gp)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.s.DeleteGeneration(e.ctx, core.System, e.project, gid); !errors.Is(err, ErrBadState) {
		t.Fatalf("delete of a running video: %v", err)
	}
}

func TestQuotas(t *testing.T) {
	e := newHDEnv(t)
	uid := e.member("quota-user", "member")
	member := &api.AdminCtx{Session: &auth.Session{User: auth.User{ID: uid, Role: auth.RoleMember}}, ProjectID: e.project}
	operator := &api.AdminCtx{Session: &auth.Session{User: auth.User{ID: uid, Role: auth.RoleOperator}}, ProjectID: e.project}

	if msg := e.s.checkQuota(e.ctx, member, 600, true); msg != "" {
		t.Fatalf("10 minutes is within the default: %q", msg)
	}
	if msg := e.s.checkQuota(e.ctx, member, 601, true); !strings.Contains(msg, "10 minutes") {
		t.Fatalf("members are limited to 10 minutes by default: %q", msg)
	}
	if msg := e.s.checkQuota(e.ctx, operator, 1800, true); msg != "" {
		t.Fatalf("operators are exempt from the length limit: %q", msg)
	}
	ps := projects.New(e.s.d)
	if err := ps.SetSettings(e.ctx, core.System, e.project, projects.Settings{MemberMaxVideoSeconds: 60, MaxConcurrentLongVideos: 1, MaxQueuedJobs: 2,
		IntermediateRetentionHours: 24}); err != nil {
		t.Fatal(err)
	}
	if msg := e.s.checkQuota(e.ctx, member, 61, true); !strings.Contains(msg, "1 minute") {
		t.Fatalf("custom limit not applied: %q", msg)
	}
	gp := genParams{Prompt: "x", Seed: 1, Width: 1280, Height: 720, FPS: 30, Seconds: 20, SegFrame: segmentFrames(8, 30), Frames: 601,
		Inputs: []string{newUUID(), newUUID(), newUUID()}}
	if _, _, err := e.s.createGeneration(e.ctx, e.project, newUUID(), &uid, gp); err != nil {
		t.Fatal(err)
	}
	if msg := e.s.checkLongQuota(e.ctx, member); !strings.Contains(msg, "long videos running") {
		t.Fatalf("one running long video reaches the limit of 1: %q", msg)
	}
	if msg := e.s.checkQuota(e.ctx, member, 10, true); msg != "" {
		t.Fatalf("1 of 2 unfinished videos is fine: %q", msg)
	}
	if _, _, err := e.s.createGeneration(e.ctx, e.project, newUUID(), &uid, gp); err != nil {
		t.Fatal(err)
	}
	if msg := e.s.checkQuota(e.ctx, member, 10, true); !strings.Contains(msg, "unfinished videos") {
		t.Fatalf("2 of 2 unfinished videos must be refused: %q", msg)
	}
	// Another user is not affected.
	other := &api.AdminCtx{Session: &auth.Session{User: auth.User{ID: e.member("quota-two", "member"), Role: auth.RoleMember}}, ProjectID: e.project}
	if msg := e.s.checkQuota(e.ctx, other, 10, true); msg != "" {
		t.Fatalf("limits are per user: %q", msg)
	}
}
