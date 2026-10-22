package pyexec

import "time"

// CodeStalled is the failure code for a task killed by the stall watchdog. It is
// retryable: the job goes back to the queue, usually to another attempt/worker.
const CodeStalled = "STALLED"

// stallMinutes is how long a Python task may stay completely silent (no
// progress, log or any other output line) before its process tree is killed.
// Job types that are not listed (controller-side aggregation and validation)
// have no watchdog unless Runner.StallTimeout is set.
var stallMinutes = map[string]int{
	"ai.generate":       30,
	"ai.draft":          30,
	"ai.refine_segment": 30,
	"ai.stitch":         30,
	"ai.evaluate":       30,
	"ai.hd_eval":        30,
	"ai.dataset_shard":  20,
	"ai.hd_encode":      20,
	"ai.local_train":    45,
	"ai.hd_train":       45,
}

// StallTimeoutFor returns the default stall limit for a job type (0 = none).
func StallTimeoutFor(jobType string) time.Duration {
	return time.Duration(stallMinutes[jobType]) * time.Minute
}
