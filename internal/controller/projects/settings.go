package projects

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
)

// Settings are the per-project limits and retention edited by administrators.
type Settings struct {
	// MemberMaxVideoSeconds is the longest video a member may request.
	// Operators and administrators are exempt from this length limit.
	MemberMaxVideoSeconds int `json:"member_max_video_seconds"`
	// MaxConcurrentLongVideos is how many long videos one user may have running.
	MaxConcurrentLongVideos int `json:"max_concurrent_long_videos"`
	// MaxQueuedJobs is how many unfinished jobs one user may have in total.
	MaxQueuedJobs int `json:"max_queued_jobs"`
	// IntermediateRetentionHours: draft and segment videos of a finished long
	// video are deleted this many hours after it completes (0 = immediately).
	IntermediateRetentionHours int `json:"intermediate_retention_hours"`
}

// DefaultSettings are used until an administrator changes them.
func DefaultSettings() Settings {
	return Settings{MemberMaxVideoSeconds: 600, MaxConcurrentLongVideos: 2, MaxQueuedJobs: 50, IntermediateRetentionHours: 24}
}

// Validate checks the ranges the database also enforces.
func (s Settings) Validate() error {
	switch {
	case s.MemberMaxVideoSeconds < 1 || s.MemberMaxVideoSeconds > 86400:
		return fmt.Errorf("member_max_video_seconds must be 1..86400")
	case s.MaxConcurrentLongVideos < 1 || s.MaxConcurrentLongVideos > 1000:
		return fmt.Errorf("max_concurrent_long_videos must be 1..1000")
	case s.MaxQueuedJobs < 1 || s.MaxQueuedJobs > 100000:
		return fmt.Errorf("max_queued_jobs must be 1..100000")
	case s.IntermediateRetentionHours < 0 || s.IntermediateRetentionHours > 8760:
		return fmt.Errorf("intermediate_retention_hours must be 0..8760")
	}
	return nil
}

// GetSettings returns a project's settings, or the defaults when none were saved.
func (s *Service) GetSettings(ctx context.Context, projectID string) (Settings, error) {
	out := DefaultSettings()
	err := s.d.DB.QueryRow(ctx, `SELECT member_max_video_seconds, max_concurrent_long_videos, max_queued_jobs, intermediate_retention_hours
		FROM project_settings WHERE project_id=$1`, projectID).
		Scan(&out.MemberMaxVideoSeconds, &out.MaxConcurrentLongVideos, &out.MaxQueuedJobs, &out.IntermediateRetentionHours)
	if err != nil && !errors.Is(err, pgx.ErrNoRows) {
		return out, err
	}
	return out, nil
}

// SetSettings validates and saves a project's settings.
func (s *Service) SetSettings(ctx context.Context, actor core.Actor, projectID string, in Settings) error {
	if err := in.Validate(); err != nil {
		return err
	}
	var by any
	if actor.Type == "user" {
		by = actor.ID
	}
	_, err := s.d.DB.Exec(ctx, `INSERT INTO project_settings (project_id, member_max_video_seconds, max_concurrent_long_videos,
			max_queued_jobs, intermediate_retention_hours, updated_by) VALUES ($1,$2,$3,$4,$5,$6)
		ON CONFLICT (project_id) DO UPDATE SET member_max_video_seconds=$2, max_concurrent_long_videos=$3,
			max_queued_jobs=$4, intermediate_retention_hours=$5, updated_by=$6, updated_at=now()`,
		projectID, in.MemberMaxVideoSeconds, in.MaxConcurrentLongVideos, in.MaxQueuedJobs, in.IntermediateRetentionHours, by)
	if err == nil {
		s.d.Audit(ctx, nil, actor, "project.settings", "project:"+projectID, "success", map[string]any{
			"member_max_video_seconds": in.MemberMaxVideoSeconds, "max_concurrent_long_videos": in.MaxConcurrentLongVideos,
			"max_queued_jobs": in.MaxQueuedJobs, "intermediate_retention_hours": in.IntermediateRetentionHours})
	}
	return err
}

// Usage is what one user currently has in flight.
type Usage struct {
	// RunningLongVideos counts the user's generations that are still running.
	RunningLongVideos int `json:"running_long_videos"`
	// QueuedVideos counts unfinished single-clip videos plus running long videos.
	QueuedVideos int `json:"queued_jobs"`
}

// UserUsage measures a user's unfinished video work in a project.
func (s *Service) UserUsage(ctx context.Context, projectID, userID string) (Usage, error) {
	var u Usage
	err := s.d.DB.QueryRow(ctx, `SELECT
			(SELECT count(*) FROM generations WHERE project_id::text=$1 AND created_by::text=$2 AND status='running'),
			(SELECT count(*) FROM jobs WHERE project_id::text=$1 AND created_by::text=$2 AND job_type='ai.generate'
				AND status IN ('ready','leased','running','uploading','validating'))`, projectID, userID).
		Scan(&u.RunningLongVideos, &u.QueuedVideos)
	u.QueuedVideos += u.RunningLongVideos
	return u, err
}
