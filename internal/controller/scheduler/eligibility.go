package scheduler

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// deviceCaps is what eligibility needs to know about an enrolled computer.
type deviceCaps struct {
	online, paused, trusted bool
	os                      string
	executors               []string
	ramMB, vramMB           int64
	busy                    bool
	rejected                bool
}

// WaitingReason explains in plain words why a ready job is not starting. It
// returns "" when some online computer could take it (the job is just queued)
// or when the reason cannot be determined.
func (s *Scheduler) WaitingReason(ctx context.Context, projectID, jobID string) string {
	var jobType, reqOS string
	var ram, vram int
	var gpu, trusted bool
	var os *string
	var retryIn float64
	if err := s.d.DB.QueryRow(ctx, `SELECT job_type, req_ram_mb, req_vram_mb, req_gpu, req_os, req_trusted,
			GREATEST(COALESCE(EXTRACT(EPOCH FROM (available_at - now())), 0), 0) FROM jobs
		WHERE id=$1 AND project_id=$2 AND status='ready'`, jobID, projectID).Scan(&jobType, &ram, &vram, &gpu, &os, &trusted, &retryIn); err != nil {
		return ""
	}
	if retryIn > 0 {
		return fmt.Sprintf("A computer had a problem with this step. It will be tried again in about %d s.", int(retryIn)+1)
	}
	if os != nil {
		reqOS = *os
	}
	disconnectAfter := time.Duration(s.d.Cfg.HeartbeatIntervalSec*s.d.Cfg.DisconnectAfterMissed) * time.Second
	rows, err := s.d.DB.Query(ctx, `SELECT d.last_seen_at > now() - make_interval(secs => $2), d.paused, d.trusted, d.os,
			COALESCE(c.executors, '{}'), COALESCE(c.ram_budget_mb, 0), COALESCE(c.vram_budget_mb, 0),
			EXISTS (SELECT 1 FROM job_attempts a WHERE a.device_id=d.id AND a.state IN ('leased','running','uploading')),
			EXISTS (SELECT 1 FROM job_attempts x WHERE x.job_id=$3 AND x.device_id=d.id AND x.state='rejected')
		FROM devices d LEFT JOIN worker_capabilities c ON c.device_id=d.id
		WHERE d.project_id=$1 AND d.status='active' LIMIT 2000`, projectID, disconnectAfter.Seconds(), jobID)
	if err != nil {
		return ""
	}
	defer rows.Close()
	var devs []deviceCaps
	for rows.Next() {
		var d deviceCaps
		var online *bool
		if err := rows.Scan(&online, &d.paused, &d.trusted, &d.os, &d.executors, &d.ramMB, &d.vramMB, &d.busy, &d.rejected); err != nil {
			return ""
		}
		d.online = online != nil && *online
		devs = append(devs, d)
	}
	return explainWaiting(jobType, ram, vram, gpu, reqOS, trusted, devs)
}

func gb(mb int64) string {
	return fmt.Sprintf("%.1f GB", float64(mb)/1024)
}

// explainWaiting narrows the fleet step by step and names the first
// requirement nobody meets.
func explainWaiting(jobType string, ramMB, vramMB int, gpu bool, reqOS string, needTrusted bool, devs []deviceCaps) string {
	if len(devs) == 0 {
		return "No computer has joined yet. Start Nivra on a computer that should do the work."
	}
	var online []deviceCaps
	for _, d := range devs {
		if d.online {
			online = append(online, d)
		}
	}
	if len(online) == 0 {
		return "No helper computer is online right now. This starts as soon as one connects."
	}
	var cands []deviceCaps
	for _, d := range online {
		if !d.paused && !d.rejected {
			cands = append(cands, d)
		}
	}
	if len(cands) == 0 {
		return "Every online computer is paused or has already declined this job. Resume sharing on one of them."
	}
	step := func(keep func(deviceCaps) bool) []deviceCaps {
		var out []deviceCaps
		for _, d := range cands {
			if keep(d) {
				out = append(out, d)
			}
		}
		return out
	}
	next := step(func(d deviceCaps) bool {
		for _, e := range d.executors {
			if e == jobType {
				return true
			}
		}
		return false
	})
	if len(next) == 0 {
		return "None of the online computers can make this kind of video (" + jobType + "). " +
			"They may be missing the AI runtime or ffmpeg; check the Workers page."
	}
	cands = next
	if vramMB > 0 || gpu {
		var best int64
		for _, d := range cands {
			best = max(best, d.vramMB)
		}
		if (vramMB > 0 && best < int64(vramMB)) || (gpu && best == 0) {
			return fmt.Sprintf("No online computer has enough graphics memory: this needs %s and the best has %s.", gb(int64(vramMB)), gb(best))
		}
		cands = step(func(d deviceCaps) bool { return d.vramMB >= int64(vramMB) && (!gpu || d.vramMB > 0) })
	}
	if ramMB > 0 {
		var best int64
		for _, d := range cands {
			best = max(best, d.ramMB)
		}
		if best < int64(ramMB) {
			return fmt.Sprintf("No online computer has enough memory: this needs %s of RAM and the best offers %s.", gb(int64(ramMB)), gb(best))
		}
		cands = step(func(d deviceCaps) bool { return d.ramMB >= int64(ramMB) })
	}
	if reqOS != "" {
		cands = step(func(d deviceCaps) bool { return strings.EqualFold(d.os, reqOS) })
		if len(cands) == 0 {
			return "No online computer runs " + reqOS + ", which this needs."
		}
	}
	if needTrusted {
		cands = step(func(d deviceCaps) bool { return d.trusted })
		if len(cands) == 0 {
			return "This needs a computer marked as trusted by an administrator, and none is online."
		}
	}
	idle := 0
	for _, d := range cands {
		if !d.busy {
			idle++
		}
	}
	if idle == 0 {
		return fmt.Sprintf("All %d matching computers are busy right now. This is next in line.", len(cands))
	}
	return ""
}
