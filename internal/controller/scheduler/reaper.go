package scheduler

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"

	"nivra/internal/controller/core"
)

// reaperLockKey ensures only one reaper runs even if two controllers share a database.
const reaperLockKey int64 = 0x4e49565241 + 1

// ReapOnce expires lapsed leases, fails jobs past their deadline and expires
// stale commands. It returns the number of attempts expired.
func (s *Scheduler) ReapOnce(ctx context.Context) (int, error) {
	expired := 0
	type jobRef struct{ project, job string }
	var touched []jobRef
	err := core.WithTx(ctx, s.d.DB, func(tx pgx.Tx) error {
		var got bool
		if err := tx.QueryRow(ctx, `SELECT pg_try_advisory_xact_lock($1)`, reaperLockKey).Scan(&got); err != nil || !got {
			return err
		}
		rows, err := tx.Query(ctx, `UPDATE job_attempts SET state='expired', completed_at=now(), error='lease expired'
			WHERE state = ANY($1) AND lease_expires_at < now()
			RETURNING id, job_id`, activeAttemptStates)
		if err != nil {
			return err
		}
		type pair struct{ attempt, job string }
		var ps []pair
		for rows.Next() {
			var p pair
			if err := rows.Scan(&p.attempt, &p.job); err != nil {
				rows.Close()
				return err
			}
			ps = append(ps, p)
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return err
		}
		for _, p := range ps {
			var project string
			// An expired lease usually means the computer slept or went offline: refund the
			// attempt (capped per job) so a laptop that closes does not fail a long video.
			err := tx.QueryRow(ctx, `UPDATE jobs SET
					status = CASE WHEN refund_count < `+maxRefundsSQL+` OR attempt_count < max_attempts THEN 'ready' ELSE 'failed' END,
					completed_at = CASE WHEN refund_count < `+maxRefundsSQL+` OR attempt_count < max_attempts THEN NULL ELSE now() END,
					attempt_count = CASE WHEN refund_count < `+maxRefundsSQL+` THEN GREATEST(attempt_count-1, 0) ELSE attempt_count END,
					refund_count = refund_count + 1,
					failure_reason = 'lease expired', current_attempt_id = NULL, updated_at = now()
				WHERE id=$1 AND current_attempt_id=$2 RETURNING project_id`, p.job, p.attempt).Scan(&project)
			if err == nil {
				touched = append(touched, jobRef{project, p.job})
			} else if err != pgx.ErrNoRows {
				return err
			}
		}
		expired = len(ps)
		drows, err := tx.Query(ctx, `UPDATE jobs SET status='failed', failure_reason='deadline exceeded', completed_at=now(), updated_at=now()
			WHERE status='ready' AND deadline_at IS NOT NULL AND deadline_at < now() RETURNING project_id, id`)
		if err != nil {
			return err
		}
		for drows.Next() {
			var r jobRef
			if err := drows.Scan(&r.project, &r.job); err != nil {
				drows.Close()
				return err
			}
			touched = append(touched, r)
		}
		drows.Close()
		_, err = tx.Exec(ctx, `UPDATE worker_commands SET state='expired', finished_at=now()
			WHERE state IN ('queued', 'delivered') AND expires_at < now()`)
		return err
	})
	if err != nil {
		return 0, err
	}
	notified := map[string]bool{}
	for _, r := range touched {
		if !notified[r.project] {
			s.d.Events.NotifyJobs(r.project)
			notified[r.project] = true
		}
		s.publishJob(ctx, r.project, r.job)
	}
	return expired, nil
}

// RunReaper runs ReapOnce periodically until ctx is cancelled.
func (s *Scheduler) RunReaper(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		if n, err := s.ReapOnce(ctx); err != nil && ctx.Err() == nil {
			s.d.Log.Warn("reaper", "err", err)
		} else if n > 0 {
			s.d.Log.Info("expired leases", "count", n)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}
