package scheduler

import (
	"context"

	"github.com/jackc/pgx/v5"
)

// TrainingShareEvery is the reserved share for training work: when a device
// qualifies for a ready training job and none of the project's last
// TrainingShareEvery-1 leases was a training job, training goes first. Under
// a flood of videos training therefore still gets at least one lease in N.
const TrainingShareEvery = 4

// pickQuery ranks ready jobs the device may run:
//
//  1. priority band (priority / 10; interactive video work shares band 2, so
//     the stages of one huge video cannot outrank other users' quick clips),
//  2. the creator with the fewest running attempts (fair share between users),
//  3. priority inside the band (finish stitch before refine before draft),
//  4. age.
//
// FOR UPDATE SKIP LOCKED keeps concurrent leases safe.
const pickQuery = `
WITH load AS (
	SELECT created_by, count(*) AS n FROM jobs
	WHERE project_id=$1 AND status IN ('leased','running','uploading') GROUP BY created_by
)
SELECT j.id FROM jobs j LEFT JOIN load l ON l.created_by IS NOT DISTINCT FROM j.created_by
WHERE j.project_id=$1 AND j.status='ready'
  AND j.job_type = ANY($2)
  AND j.req_ram_mb <= $3 AND j.req_vram_mb <= $4 AND (NOT j.req_gpu OR $4 > 0)
  AND (j.req_os IS NULL OR j.req_os = $5)
  AND (j.deadline_at IS NULL OR j.deadline_at > now())
  AND (j.available_at IS NULL OR j.available_at <= now())
  AND NOT EXISTS (SELECT 1 FROM job_attempts x WHERE x.job_id=j.id AND x.device_id=$6 AND x.state='rejected')
  AND (NOT j.req_trusted OR $7)
  AND (NOT $8 OR j.category='training')
ORDER BY floor(j.priority / 10.0) DESC, COALESCE(l.n, 0), j.priority DESC, j.created_at
LIMIT 1 FOR UPDATE OF j SKIP LOCKED`

// pickJob selects the job to lease to a device (pgx.ErrNoRows when none).
func (s *Scheduler) pickJob(ctx context.Context, tx pgx.Tx, projectID, deviceID string, executors []string, ram, vram int64,
	devOS string, trusted bool) (string, error) {
	var jobID string
	if s.preferTraining(ctx, tx, projectID) {
		err := tx.QueryRow(ctx, pickQuery, projectID, executors, ram, vram, devOS, deviceID, trusted, true).Scan(&jobID)
		if err == nil {
			return jobID, nil
		}
		if err != pgx.ErrNoRows {
			return "", err
		}
	}
	err := tx.QueryRow(ctx, pickQuery, projectID, executors, ram, vram, devOS, deviceID, trusted, false).Scan(&jobID)
	return jobID, err
}

// preferTraining reports whether the reserved training share is due: none of
// the project's most recent TrainingShareEvery-1 leases was a training job.
func (s *Scheduler) preferTraining(ctx context.Context, tx pgx.Tx, projectID string) bool {
	var recentTraining bool
	err := tx.QueryRow(ctx, `SELECT COALESCE(bool_or(x.category='training'), false) FROM (
			SELECT j.category FROM job_attempts a JOIN jobs j ON j.id=a.job_id
			WHERE j.project_id=$1 ORDER BY a.leased_at DESC LIMIT $2) x`, projectID, TrainingShareEvery-1).Scan(&recentTraining)
	return err == nil && !recentTraining
}
