package training

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
	"nivra/internal/controller/scheduler"
	"nivra/internal/shared/longvideo"
	"nivra/internal/shared/protocol"
)

// Long videos (docs/architecture/LONG_VIDEO.md): a generation is one draft
// job, then N refine jobs that any workers run in parallel, then a stitch
// job. advanceGeneration moves a generation forward; it is idempotent and runs
// both when a child job is finalised and from the periodic tick, so a crash
// between commit and hook never strands a generation.

// Scheduling priorities of the video stages. All of them sit in priority band 2
// (priority / 10) together with single clips (20), so one user's huge video
// cannot outrank other users' quick videos; inside a user's own work the later
// stage goes first. See scheduler/fair.go.
const (
	prioGenerate = 20
	prioDraft    = 22
	prioRefine   = 24
	prioStitch   = 26

	// A crash, a closing laptop or a stale lease burns an attempt, so the stages
	// of a long video get more of them than the default 3.
	attemptsDraft  = 4
	attemptsRefine = 6
)

// needsAttentionTTL is how long a long video with a failed part waits for its
// owner to regenerate the part before the video is failed for good.
const needsAttentionTTL = 72 * time.Hour

// genParams are the user's settings, stored on the generation.
type genParams struct {
	Prompt   string   `json:"prompt"`
	Seed     int      `json:"seed"`
	Style    string   `json:"style,omitempty"`
	Sampler  string   `json:"sampler,omitempty"`
	Steps    int      `json:"steps,omitempty"`
	CFGScale float64  `json:"cfg_scale,omitempty"`
	Width    int      `json:"width"`
	Height   int      `json:"height"`
	FPS      int      `json:"fps"`
	Seconds  float64  `json:"seconds"`
	SegFrame int      `json:"segment_frames"`
	Frames   int      `json:"frames"`
	Inputs   []string `json:"inputs"` // weights, manifest, vae, [text encoder archive] artifact ids
	RAMMB    int      `json:"ram_mb"`
	VRAMMB   int      `json:"vram_mb"`
}

// segmentFrames and segmentPlan come from the shared long-video plan.
func segmentFrames(seconds float64, fps int) int { return longvideo.SegmentFrames(seconds, fps) }

func segmentPlan(frames, segFrames int) [][2]int { return longvideo.Plan(frames, segFrames) }

func (p genParams) inputs(extra ...protocol.ArtifactRef) []protocol.ArtifactRef {
	refs := []protocol.ArtifactRef{{ID: p.Inputs[0], Role: "weights"}, {ID: p.Inputs[1], Role: "manifest"}, {ID: p.Inputs[2], Role: "vae_weights"}}
	if len(p.Inputs) > 3 && p.Inputs[3] != "" { // generations stored before 0009 have exactly three
		refs = append(refs, protocol.ArtifactRef{ID: p.Inputs[3], Role: "text_encoder_archive"})
	}
	return append(refs, extra...)
}

// artifactIDs lists the ids of versionInputs refs in order.
func artifactIDs(refs []protocol.ArtifactRef) []string {
	ids := make([]string, len(refs))
	for i, r := range refs {
		ids[i] = r.ID
	}
	return ids
}

func (p genParams) common() map[string]any {
	m := map[string]any{"prompt": p.Prompt, "fps": p.FPS}
	if p.Style != "" {
		m["style"] = p.Style
	}
	if p.Sampler != "" {
		m["sampler"] = p.Sampler
	}
	if p.Steps > 0 {
		m["steps"] = p.Steps
	}
	if p.CFGScale > 0 {
		m["cfg_scale"] = p.CFGScale
	}
	return m
}

// createGeneration stores a generation and enqueues its draft job.
func (s *Service) createGeneration(ctx context.Context, projectID, versionID string, createdBy *string, p genParams) (genID, draftJob string, err error) {
	plan := segmentPlan(p.Frames, p.SegFrame)
	err = core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		pb, _ := json.Marshal(p)
		if err := tx.QueryRow(ctx, `INSERT INTO generations (project_id, model_version, created_by, params, segments)
			VALUES ($1,$2,$3,$4,$5) RETURNING id`, projectID, versionID, createdBy, pb, len(plan)).Scan(&genID); err != nil {
			return err
		}
		params := p.common()
		params["seed"], params["seconds"], params["name"] = p.Seed, p.Seconds, "draft"
		params["segment_frames"] = p.SegFrame
		jb, _ := json.Marshal(params)
		// The draft is low resolution: the model's requirements still apply
		// (it loads the same weights).
		draftJob, err = s.sched.Create(ctx, tx, scheduler.NewJob{ProjectID: projectID, JobType: "ai.draft", Params: jb,
			Inputs: p.inputs(), ModelVersion: &versionID, Priority: prioDraft, MaxAttempts: attemptsDraft, ReqRAMMB: p.RAMMB, ReqVRAMMB: p.VRAMMB,
			ReqGPU: p.VRAMMB > 0, LeaseSeconds: 900, CreatedBy: createdBy, IdempotencyKey: "gen:" + genID + ":draft:0"})
		if err != nil {
			return err
		}
		_, err = tx.Exec(ctx, `INSERT INTO generation_jobs (generation_id, stage, idx, job_id) VALUES ($1,'draft',0,$2)`, genID, draftJob)
		return err
	})
	if err == nil {
		s.d.Events.NotifyJobs(projectID)
	}
	return genID, draftJob, err
}

type childJob struct {
	stage    string
	idx      int
	jobID    string
	status   string
	failure  string
	progress float64
	device   string
}

func (s *Service) genChildren(ctx context.Context, q core.Querier, genID string) ([]childJob, error) {
	rows, err := q.Query(ctx, `SELECT g.stage, g.idx, g.job_id::text, j.status, j.failure_reason,
			COALESCE(a.progress, 0), COALESCE(d.name, '')
		FROM generation_jobs g JOIN jobs j ON j.id = g.job_id
		LEFT JOIN job_attempts a ON a.id = j.current_attempt_id
		LEFT JOIN devices d ON d.id = a.device_id
		WHERE g.generation_id = $1 ORDER BY g.stage, g.idx`, genID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []childJob
	for rows.Next() {
		var c childJob
		if err := rows.Scan(&c.stage, &c.idx, &c.jobID, &c.status, &c.failure, &c.progress, &c.device); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

// resultArtifact returns the artifact of a given kind from a job's accepted result.
func resultArtifact(ctx context.Context, q core.Querier, jobID, kind string) (string, error) {
	var id string
	err := q.QueryRow(ctx, `SELECT ar.id::text FROM artifacts ar
		JOIN job_attempts a ON a.id = ar.attempt_id
		WHERE a.job_id = $1 AND a.state = 'accepted' AND ar.kind = $2 AND ar.state = 'complete'
		ORDER BY ar.created_at DESC LIMIT 1`, jobID, kind).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", fmt.Errorf("job %s has no accepted %s artifact", jobID, kind)
	}
	return id, err
}

// advanceGeneration applies every transition that is due. Returns child jobs
// to cancel (done after commit through the scheduler).
func (s *Service) advanceGeneration(ctx context.Context, genID string) error {
	var cancel []string
	var projectID string
	created := false
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var status, stage string
		var segments int
		var raw []byte
		var versionID string
		var createdBy *string
		var attentionSince *time.Time
		if err := tx.QueryRow(ctx, `SELECT project_id::text, status, stage, segments, params, model_version::text, created_by::text, needs_attention_since
			FROM generations WHERE id=$1 FOR UPDATE`, genID).Scan(&projectID, &status, &stage, &segments, &raw, &versionID, &createdBy, &attentionSince); err != nil {
			return err
		}
		if status != "running" {
			return nil
		}
		var p genParams
		if err := json.Unmarshal(raw, &p); err != nil {
			return err
		}
		kids, err := s.genChildren(ctx, tx, genID)
		if err != nil {
			return err
		}
		fail := func(newStatus, reason string) error {
			for _, k := range kids {
				if !scheduler.IsTerminalJob(k.status) {
					cancel = append(cancel, k.jobID)
				}
			}
			_, err := tx.Exec(ctx, `UPDATE generations SET status=$2, failure_reason=$3, updated_at=now(), completed_at=now() WHERE id=$1`,
				genID, newStatus, truncate(reason, 900))
			return err
		}
		// A part that used up all its attempts does not throw the whole video away
		// (it may be hours of work): the video stays "running" but needs attention,
		// and its owner can regenerate the part. Cancelling a part cancels the video.
		var failedKid *childJob
		for i, k := range kids {
			switch k.status {
			case scheduler.JobFailed:
				if failedKid == nil {
					failedKid = &kids[i]
				}
			case scheduler.JobCancelled:
				return fail("cancelled", "a part of this video was cancelled")
			}
		}
		if failedKid != nil {
			reason := fmt.Sprintf("%s part %d failed after several tries: %s", failedKid.stage, failedKid.idx, friendlyFailure(failedKid.failure))
			if attentionSince != nil && time.Since(*attentionSince) > needsAttentionTTL {
				return fail("failed", reason+" (nobody regenerated it within 3 days)")
			}
			_, err := tx.Exec(ctx, `UPDATE generations SET needs_attention_since=COALESCE(needs_attention_since, now()),
				failure_reason=$2, updated_at=now() WHERE id=$1`, genID, truncate(reason, 900))
			return err
		}
		if attentionSince != nil {
			if _, err := tx.Exec(ctx, `UPDATE generations SET needs_attention_since=NULL, failure_reason='', updated_at=now() WHERE id=$1`, genID); err != nil {
				return err
			}
		}
		done := func(st string) (n int, ids []string) {
			for _, k := range kids {
				if k.stage == st && k.status == scheduler.JobCompleted {
					n++
					ids = append(ids, k.jobID)
				}
			}
			return
		}
		switch stage {
		case "draft":
			n, ids := done("draft")
			if n == 0 {
				return nil
			}
			draftVideo, err := resultArtifact(ctx, tx, ids[0], "draft_video")
			if err != nil {
				return fail("failed", err.Error())
			}
			for i, seg := range segmentPlan(p.Frames, p.SegFrame) {
				params := p.common()
				params["seed"], params["width"], params["height"] = (p.Seed+i)%2147483647, p.Width, p.Height
				params["index"], params["start"], params["end"], params["name"] = i, seg[0], seg[1], fmt.Sprintf("segment_%03d", i)
				jb, _ := json.Marshal(params)
				id, err := s.sched.Create(ctx, tx, scheduler.NewJob{ProjectID: projectID, JobType: "ai.refine_segment", Params: jb,
					Inputs: p.inputs(protocol.ArtifactRef{ID: draftVideo, Role: "draft_video"}), ModelVersion: &versionID,
					Priority: prioRefine, MaxAttempts: attemptsRefine, ReqRAMMB: p.RAMMB, ReqVRAMMB: p.VRAMMB, ReqGPU: p.VRAMMB > 0, LeaseSeconds: 1800,
					CreatedBy: createdBy, IdempotencyKey: fmt.Sprintf("gen:%s:refine:%d", genID, i)})
				if err != nil {
					return err
				}
				if _, err := tx.Exec(ctx, `INSERT INTO generation_jobs (generation_id, stage, idx, job_id) VALUES ($1,'refine',$2,$3)
					ON CONFLICT DO NOTHING`, genID, i, id); err != nil {
					return err
				}
			}
			created = true
			_, err = tx.Exec(ctx, `UPDATE generations SET stage='refine', updated_at=now() WHERE id=$1`, genID)
			return err
		case "refine":
			n, _ := done("refine")
			if n < segments {
				return nil
			}
			refs := []protocol.ArtifactRef{}
			for _, k := range kids {
				if k.stage != "refine" {
					continue
				}
				art, err := resultArtifact(ctx, tx, k.jobID, "segment_video")
				if err != nil {
					return fail("failed", err.Error())
				}
				refs = append(refs, protocol.ArtifactRef{ID: art, Role: fmt.Sprintf("segment_%03d", k.idx)})
			}
			jb, _ := json.Marshal(map[string]any{"fps": p.FPS, "count": len(refs), "crossfade_frames": 4, "name": "video"})
			id, err := s.sched.Create(ctx, tx, scheduler.NewJob{ProjectID: projectID, JobType: "ai.stitch", Params: jb, Inputs: refs,
				ModelVersion: &versionID, Priority: prioStitch, MaxAttempts: attemptsRefine, ReqRAMMB: 1024, LeaseSeconds: 1800, CreatedBy: createdBy,
				IdempotencyKey: "gen:" + genID + ":stitch:0"})
			if err != nil {
				return err
			}
			if _, err := tx.Exec(ctx, `INSERT INTO generation_jobs (generation_id, stage, idx, job_id) VALUES ($1,'stitch',0,$2)
				ON CONFLICT DO NOTHING`, genID, id); err != nil {
				return err
			}
			created = true
			_, err = tx.Exec(ctx, `UPDATE generations SET stage='stitch', updated_at=now() WHERE id=$1`, genID)
			return err
		case "stitch":
			n, ids := done("stitch")
			if n == 0 {
				return nil
			}
			video, err := resultArtifact(ctx, tx, ids[0], "video")
			if err != nil {
				return fail("failed", err.Error())
			}
			_, err = tx.Exec(ctx, `UPDATE generations SET status='completed', stage='done', artifact_id=$2, updated_at=now(), completed_at=now()
				WHERE id=$1`, genID, video)
			return err
		}
		return nil
	})
	if err != nil {
		return err
	}
	for _, j := range cancel {
		if err := s.sched.Cancel(ctx, core.Actor{Type: "system", ID: "generation"}, projectID, j); err != nil &&
			!errors.Is(err, scheduler.ErrBadState) && !errors.Is(err, scheduler.ErrNotFound) {
			return err
		}
	}
	if created {
		s.d.Events.NotifyJobs(projectID)
	}
	return nil
}

// tickGenerations advances every running generation (crash recovery and
// failures detected by the lease reaper).
func (s *Service) tickGenerations(ctx context.Context) {
	rows, err := s.d.DB.Query(ctx, `SELECT id::text FROM generations WHERE status='running' ORDER BY created_at LIMIT 500`)
	if err != nil {
		return
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		return
	}
	for _, id := range ids {
		if ctx.Err() != nil {
			return
		}
		_ = s.advanceGeneration(ctx, id)
	}
}

// hdDefaults reads the bundle's default clip length and fps from the manifest.
func (s *Service) hdDefaults(ctx context.Context, versionID string) (seconds float64, fps int) {
	seconds, fps = 8, 30
	var raw []byte
	if s.d.DB.QueryRow(ctx, `SELECT manifest FROM model_versions WHERE id=$1`, versionID).Scan(&raw) != nil {
		return
	}
	var m struct {
		Seconds float64 `json:"default_seconds"`
		FPS     float64 `json:"default_fps"`
	}
	if json.Unmarshal(raw, &m) == nil {
		if m.Seconds > 0 {
			seconds = m.Seconds
		}
		if m.FPS > 0 {
			fps = int(m.FPS)
		}
	}
	return
}
