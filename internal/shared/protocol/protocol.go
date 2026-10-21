// Package protocol defines the versioned wire contract between Nivra workers
// and the Nivra controller (protocol "nivra.worker/1").
package protocol

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Version is the worker protocol version. Incompatible changes bump it.
const Version = 1

// MinSupportedVersion is the oldest worker protocol version the controller accepts.
const MinSupportedVersion = 1

// HeaderProtocol carries the protocol version on requests without a body.
const HeaderProtocol = "X-Nivra-Protocol"

// MaxMessageBytes bounds every JSON request body on worker routes.
const MaxMessageBytes = 256 << 10

// Envelope wraps every worker POST body. Identity fields are verified against
// the authenticated session; the payload is decoded per message type.
type Envelope struct {
	ProtocolVersion int             `json:"protocol_version"`
	MessageID       string          `json:"message_id"`
	MessageType     string          `json:"message_type"`
	ProjectID       string          `json:"project_id,omitempty"`
	DeviceID        string          `json:"device_id,omitempty"`
	Timestamp       int64           `json:"timestamp_ms"`
	Payload         json.RawMessage `json:"payload"`
}

// Message types.
const (
	MsgEnrollBegin    = "enroll.begin"
	MsgEnrollComplete = "enroll.complete"
	MsgSession        = "session.open"
	MsgHeartbeat      = "heartbeat"
	MsgJobAck         = "job.ack"
	MsgJobProgress    = "job.progress"
	MsgJobComplete    = "job.complete"
	MsgJobFail        = "job.fail"
	MsgCommandAck     = "command.ack"
	MsgUploadBegin    = "upload.begin"
	MsgLogBatch       = "log.batch"
)

// CheckVersion rejects incompatible protocol versions.
func CheckVersion(v int) error {
	if v < MinSupportedVersion || v > Version {
		return fmt.Errorf("unsupported protocol version %d (supported %d..%d)", v, MinSupportedVersion, Version)
	}
	return nil
}

// ErrorBody is the JSON error shape returned by every API route.
type ErrorBody struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is a machine-readable code plus a human message.
type ErrorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Error codes shared by controller and worker.
const (
	ErrBadRequest      = "bad_request"
	ErrUnauthorized    = "unauthorized"
	ErrForbidden       = "forbidden"
	ErrRevoked         = "device_revoked"
	ErrNotFound        = "not_found"
	ErrConflict        = "conflict"
	ErrStaleLease      = "stale_lease"
	ErrRateLimited     = "rate_limited"
	ErrTooLarge        = "too_large"
	ErrUnsupported     = "unsupported_protocol"
	ErrInternal        = "internal"
	ErrUnavailable     = "unavailable"
	ErrInvalidCode     = "invalid_code"
	ErrValidation      = "validation_failed"
	ErrIncompatible    = "incompatible"
	ErrNotEnrolled     = "not_enrolled"
	ErrPaused          = "paused"
	ErrChecksum        = "checksum_mismatch"
	ErrOffsetMismatch  = "offset_mismatch"
	ErrAlreadyComplete = "already_complete"
)

// APIError is returned by clients when the controller answers with an error body.
type APIError struct {
	Status  int
	Code    string
	Message string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("controller error %d %s: %s", e.Status, e.Code, e.Message)
}

// IsCode reports whether err is an APIError with the given code.
func IsCode(err error, code string) bool {
	var ae *APIError
	return errors.As(err, &ae) && ae.Code == code
}

// DeviceInfo describes a worker at enrollment time.
type DeviceInfo struct {
	Name            string `json:"name"`
	ContributorName string `json:"contributor_name,omitempty"`
	OS              string `json:"os"`
	Arch            string `json:"arch"`
	AgentVersion    string `json:"agent_version"`
}

// EnrollBeginRequest starts enrollment with a project code and the worker's
// freshly generated Ed25519 public key.
type EnrollBeginRequest struct {
	Code      string     `json:"code"`
	PublicKey []byte     `json:"public_key"`
	Device    DeviceInfo `json:"device"`
}

// EnrollBeginResponse carries a challenge plus the controller identity proof.
// ControllerSignature signs EnrollChallengeMessage(...) with ControllerKey.
type EnrollBeginResponse struct {
	EnrollmentID        string    `json:"enrollment_id"`
	Nonce               []byte    `json:"nonce"`
	ProjectID           string    `json:"project_id"`
	ProjectName         string    `json:"project_name"`
	ControllerKey       []byte    `json:"controller_key"`
	ControllerSignature []byte    `json:"controller_signature"`
	ExpiresAt           time.Time `json:"expires_at"`
}

// EnrollCompleteRequest proves possession of the device private key.
type EnrollCompleteRequest struct {
	EnrollmentID string `json:"enrollment_id"`
	Signature    []byte `json:"signature"`
}

// EnrollCompleteResponse confirms registration.
type EnrollCompleteResponse struct {
	DeviceID    string `json:"device_id"`
	ProjectID   string `json:"project_id"`
	ProjectName string `json:"project_name"`
	DeviceName  string `json:"device_name"`
}

// SessionRequest exchanges a device signature for a short-lived bearer token.
type SessionRequest struct {
	DeviceID    string `json:"device_id"`
	TimestampMS int64  `json:"timestamp_ms"`
	Nonce       []byte `json:"nonce"`
	Signature   []byte `json:"signature"`
}

// SessionResponse returns the bearer token and a controller identity proof
// over SessionAckMessage(...), letting the worker re-verify its pinned key.
type SessionResponse struct {
	SessionID            string    `json:"session_id"`
	Token                string    `json:"token"`
	ExpiresAt            time.Time `json:"expires_at"`
	ServerTime           time.Time `json:"server_time"`
	HeartbeatIntervalSec int       `json:"heartbeat_interval_sec"`
	ProjectID            string    `json:"project_id"`
	ProjectName          string    `json:"project_name"`
	DeviceName           string    `json:"device_name"`
	ControllerSignature  []byte    `json:"controller_signature"`
}

// Signing domains. Every signed message is domain separated.
func EnrollChallengeMessage(enrollmentID string, nonce, devicePub []byte, projectID string) []byte {
	return []byte(fmt.Sprintf("nivra-enroll-challenge-v1|%s|%x|%x|%s", enrollmentID, nonce, devicePub, projectID))
}

// EnrollProofMessage is what the device signs to complete enrollment.
func EnrollProofMessage(enrollmentID string, nonce []byte, projectID string) []byte {
	return []byte(fmt.Sprintf("nivra-enroll-proof-v1|%s|%x|%s", enrollmentID, nonce, projectID))
}

// SessionMessage is what the device signs to open a session.
func SessionMessage(deviceID string, timestampMS int64, nonce []byte) []byte {
	return []byte(fmt.Sprintf("nivra-session-v1|%s|%d|%x", deviceID, timestampMS, nonce))
}

// SessionAckMessage is what the controller signs when issuing a session.
func SessionAckMessage(deviceID string, nonce []byte, sessionID string) []byte {
	return []byte(fmt.Sprintf("nivra-session-ack-v1|%s|%x|%s", deviceID, nonce, sessionID))
}

// CPUInfo describes the host CPU.
type CPUInfo struct {
	Model   string `json:"model"`
	Cores   int    `json:"cores"`
	Threads int    `json:"threads"`
}

// GPUInfo describes one accelerator.
type GPUInfo struct {
	Index         int    `json:"index"`
	Name          string `json:"name"`
	VRAMMB        int64  `json:"vram_mb"`
	Driver        string `json:"driver,omitempty"`
	ComputeCap    string `json:"compute_cap,omitempty"`
	CUDAAvailable bool   `json:"cuda_available"`
}

// Capabilities is reported at session start and whenever it changes.
type Capabilities struct {
	SchemaVersion int                `json:"schema_version"`
	OS            string             `json:"os"`
	Arch          string             `json:"arch"`
	CPU           CPUInfo            `json:"cpu"`
	GPUs          []GPUInfo          `json:"gpus"`
	RAMMB         int64              `json:"ram_mb"`
	RAMBudgetMB   int64              `json:"ram_budget_mb"`
	VRAMMB        int64              `json:"vram_mb"`
	VRAMBudgetMB  int64              `json:"vram_budget_mb"`
	StorageFreeMB int64              `json:"storage_free_mb"`
	Executors     []string           `json:"executors"`
	Backends      []string           `json:"backends"`
	Benchmark     map[string]float64 `json:"benchmark,omitempty"`
}

// CapabilitiesSchemaVersion is the current Capabilities schema.
const CapabilitiesSchemaVersion = 1

// Worker states reported by the agent.
const (
	WorkerIdle     = "idle"
	WorkerBusy     = "busy"
	WorkerPaused   = "paused"
	WorkerDegraded = "degraded"
	WorkerStopping = "stopping"
)

// MetricsSnapshot is a point-in-time resource sample.
type MetricsSnapshot struct {
	CPUPct     float64 `json:"cpu_pct"`
	RAMUsedMB  int64   `json:"ram_used_mb"`
	GPUPct     float64 `json:"gpu_pct"`
	VRAMUsedMB int64   `json:"vram_used_mb"`
	NetRxKbps  float64 `json:"net_rx_kbps"`
	NetTxKbps  float64 `json:"net_tx_kbps"`
}

// AttemptStatus is the worker's view of an attempt it holds.
type AttemptStatus struct {
	AttemptID    string  `json:"attempt_id"`
	FencingToken int64   `json:"fencing_token"`
	Progress     float64 `json:"progress"`
	Stage        string  `json:"stage"`
}

// HeartbeatRequest keeps the session and leases alive.
type HeartbeatRequest struct {
	State        string          `json:"state"`
	StateDetail  string          `json:"state_detail,omitempty"`
	Attempts     []AttemptStatus `json:"attempts"`
	Metrics      MetricsSnapshot `json:"metrics"`
	Capabilities *Capabilities   `json:"capabilities,omitempty"`
	AgentVersion string          `json:"agent_version"`
}

// HeartbeatResponse tells the worker what the controller believes.
type HeartbeatResponse struct {
	ServerTime       time.Time `json:"server_time"`
	NextHeartbeatSec int       `json:"next_heartbeat_sec"`
	PendingCommands  int       `json:"pending_commands"`
	Paused           bool      `json:"paused"`
	// InvalidAttempts lists attempts the worker reported that are no longer
	// valid (expired, reassigned, cancelled). The worker must stop them.
	InvalidAttempts []string  `json:"invalid_attempts"`
	SessionExpires  time.Time `json:"session_expires"`
}

// ArtifactRef points at a downloadable artifact.
type ArtifactRef struct {
	ID     string `json:"id"`
	Role   string `json:"role"`
	Kind   string `json:"kind"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// JobAssignment is a leased job attempt.
type JobAssignment struct {
	AttemptID      string          `json:"attempt_id"`
	JobID          string          `json:"job_id"`
	JobType        string          `json:"job_type"`
	Category       string          `json:"category"`
	Params         json.RawMessage `json:"params"`
	Inputs         []ArtifactRef   `json:"inputs"`
	ModelVersion   string          `json:"model_version,omitempty"`
	FencingToken   int64           `json:"fencing_token"`
	AttemptNumber  int             `json:"attempt_number"`
	LeaseSeconds   int             `json:"lease_seconds"`
	LeaseExpiresAt time.Time       `json:"lease_expires_at"`
}

// JobAckRequest confirms the worker started the attempt.
type JobAckRequest struct {
	FencingToken int64 `json:"fencing_token"`
}

// JobProgressRequest reports progress; State may be "running" or "uploading".
type JobProgressRequest struct {
	FencingToken int64   `json:"fencing_token"`
	State        string  `json:"state"`
	Progress     float64 `json:"progress"`
	Stage        string  `json:"stage"`
	Message      string  `json:"message,omitempty"`
}

// JobCompleteRequest submits a result. Submissions are idempotent per attempt.
type JobCompleteRequest struct {
	FencingToken int64           `json:"fencing_token"`
	Result       json.RawMessage `json:"result"`
	ArtifactID   string          `json:"artifact_id,omitempty"`
	// Artifacts maps artifact kind to uploaded artifact id (all outputs).
	Artifacts  map[string]string `json:"artifacts,omitempty"`
	Checksum   string            `json:"checksum"`
	DurationMS int64             `json:"duration_ms"`
}

// Result statuses returned on completion.
const (
	ResultValidating = "validating"
	ResultAccepted   = "accepted"
	ResultRejected   = "rejected"
)

// JobCompleteResponse reports the controller's view of the result.
type JobCompleteResponse struct {
	Status    string `json:"status"`
	Reason    string `json:"reason,omitempty"`
	Duplicate bool   `json:"duplicate"`
}

// JobFailRequest reports a failed attempt.
type JobFailRequest struct {
	FencingToken int64  `json:"fencing_token"`
	Code         string `json:"code"`
	Message      string `json:"message"`
	Retryable    bool   `json:"retryable"`
}

// Worker command types.
const (
	CmdPause               = "pause"
	CmdResume              = "resume"
	CmdStopTask            = "stop_task"
	CmdDisconnect          = "disconnect"
	CmdRefreshCapabilities = "refresh_capabilities"
)

// Command states.
const (
	CmdQueued       = "queued"
	CmdDelivered    = "delivered"
	CmdAcknowledged = "acknowledged"
	CmdExecuting    = "executing"
	CmdCompleted    = "completed"
	CmdFailed       = "failed"
	CmdExpired      = "expired"
)

// Command is a reliable controller-to-worker instruction.
type Command struct {
	ID        string          `json:"id"`
	Type      string          `json:"type"`
	Payload   json.RawMessage `json:"payload"`
	State     string          `json:"state"`
	CreatedAt time.Time       `json:"created_at"`
	ExpiresAt time.Time       `json:"expires_at"`
}

// CommandAckRequest moves a command forward (acknowledged/executing/completed/failed).
type CommandAckRequest struct {
	State  string          `json:"state"`
	Result json.RawMessage `json:"result,omitempty"`
}

// UploadBeginRequest starts a resumable upload.
type UploadBeginRequest struct {
	Kind        string `json:"kind"`
	Size        int64  `json:"size"`
	SHA256      string `json:"sha256"`
	ContentType string `json:"content_type,omitempty"`
	AttemptID   string `json:"attempt_id,omitempty"`
}

// UploadStatus describes an in-progress upload.
type UploadStatus struct {
	UploadID   string `json:"upload_id"`
	Offset     int64  `json:"offset"`
	Size       int64  `json:"size"`
	ChunkSize  int64  `json:"chunk_size"`
	State      string `json:"state"`
	ArtifactID string `json:"artifact_id,omitempty"`
}

// LogLine is one worker log entry shipped to the controller.
type LogLine struct {
	TimeMS  int64             `json:"time_ms"`
	Level   string            `json:"level"`
	Service string            `json:"service"`
	Message string            `json:"message"`
	Fields  map[string]string `json:"fields,omitempty"`
}

// LogBatch carries buffered worker logs.
type LogBatch struct {
	Lines []LogLine `json:"lines"`
}

// WorkerSelf is returned by GET /api/v1/worker/me.
type WorkerSelf struct {
	DeviceID      string            `json:"device_id"`
	DeviceName    string            `json:"device_name"`
	ProjectID     string            `json:"project_id"`
	ProjectName   string            `json:"project_name"`
	Paused        bool              `json:"paused"`
	Contributions []ContributionSum `json:"contributions"`
}

// ContributionSum aggregates verified contributions by category.
type ContributionSum struct {
	Category string  `json:"category"`
	Jobs     int64   `json:"jobs"`
	Seconds  float64 `json:"seconds"`
	Units    float64 `json:"units"`
	Credit   float64 `json:"credit"`
}
