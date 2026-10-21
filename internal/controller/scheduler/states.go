package scheduler

// Job statuses (jobs.status).
const (
	JobCreated    = "created"
	JobReady      = "ready"
	JobLeased     = "leased"
	JobRunning    = "running"
	JobUploading  = "uploading"
	JobValidating = "validating"
	JobCompleted  = "completed"
	JobFailed     = "failed"
	JobCancelled  = "cancelled"
)

// Attempt states (job_attempts.state).
const (
	AttLeased     = "leased"
	AttRunning    = "running"
	AttUploading  = "uploading"
	AttValidating = "validating"
	AttAccepted   = "accepted"
	AttRejected   = "rejected"
	AttExpired    = "expired"
	AttFailed     = "failed"
	AttCancelled  = "cancelled"
)

// jobTransitions is the job state machine. "Accepted/Rejected" from the
// specification are result verdicts: an accepted result moves the job to
// completed; a rejected one returns it to ready (retry) or failed.
var jobTransitions = map[string][]string{
	JobCreated:    {JobReady, JobCancelled},
	JobReady:      {JobLeased, JobCancelled, JobFailed},
	JobLeased:     {JobRunning, JobUploading, JobValidating, JobReady, JobFailed, JobCancelled},
	JobRunning:    {JobUploading, JobValidating, JobReady, JobFailed, JobCancelled},
	JobUploading:  {JobRunning, JobValidating, JobReady, JobFailed, JobCancelled},
	JobValidating: {JobCompleted, JobReady, JobFailed, JobCancelled},
	JobCompleted:  {},
	JobFailed:     {JobReady}, // manual retry
	JobCancelled:  {},
}

// CanTransition reports whether a job may move from one status to another.
func CanTransition(from, to string) bool {
	for _, s := range jobTransitions[from] {
		if s == to {
			return true
		}
	}
	return false
}

// activeAttemptStates hold a lease that the reaper may expire.
var activeAttemptStates = []string{AttLeased, AttRunning, AttUploading}

// IsTerminalJob reports whether no further work will happen.
func IsTerminalJob(s string) bool { return s == JobCompleted || s == JobFailed || s == JobCancelled }
