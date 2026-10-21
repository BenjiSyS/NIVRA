// Package client talks to the Nivra controller API on behalf of a worker:
// enrollment, signed sessions, enveloped messages, resumable transfers.
package client

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"nivra/internal/shared/config"
	"nivra/internal/shared/protocol"
	"nivra/internal/shared/secure"
	"nivra/internal/shared/versioning"
	"nivra/internal/worker/identity"
)

// ErrControllerIdentity means the controller's signature did not match the pinned key.
var ErrControllerIdentity = errors.New("controller identity verification failed (possible impersonation); refusing to continue")

// Client is safe for concurrent use.
type Client struct {
	base *url.URL
	hc   *http.Client
	id   *identity.Identity

	mu        sync.Mutex
	refreshMu sync.Mutex // serialises session (re)opening: each new session revokes the previous one
	token     string
	expires   time.Time
	heartbeat int

	// Byte counters for network telemetry.
	rx, tx atomic.Int64
	// Limit caps transfer bandwidth in bytes/second (0 = unlimited).
	Limit atomic.Int64
}

// New creates a client for an enrolled identity.
func New(id *identity.Identity, allowInsecure bool) (*Client, error) {
	u, err := config.CheckControllerURL(id.ControllerURL, allowInsecure)
	if err != nil {
		return nil, err
	}
	return &Client{base: u, id: id, hc: &http.Client{Timeout: 0}}, nil
}

// Identity returns the identity in use.
func (c *Client) Identity() *identity.Identity { return c.id }

// Bytes returns cumulative received and sent bytes.
func (c *Client) Bytes() (rx, tx int64) { return c.rx.Load(), c.tx.Load() }

// HeartbeatInterval is the controller-provided heartbeat period.
func (c *Client) HeartbeatInterval() time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.heartbeat <= 0 {
		return 15 * time.Second
	}
	return time.Duration(c.heartbeat) * time.Second
}

func envelope(msgType, projectID, deviceID string, payload any) ([]byte, error) {
	p, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return json.Marshal(protocol.Envelope{ProtocolVersion: protocol.Version, MessageID: hex.EncodeToString(secure.RandomBytes(12)),
		MessageType: msgType, ProjectID: projectID, DeviceID: deviceID, Timestamp: time.Now().UnixMilli(), Payload: p})
}

func decodeError(resp *http.Response) error {
	var eb protocol.ErrorBody
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if json.Unmarshal(b, &eb) == nil && eb.Error.Code != "" {
		return &protocol.APIError{Status: resp.StatusCode, Code: eb.Error.Code, Message: eb.Error.Message}
	}
	return &protocol.APIError{Status: resp.StatusCode, Code: "http_" + strconv.Itoa(resp.StatusCode), Message: string(bytes.TrimSpace(b))}
}

// rawPost sends an enveloped message without a session (enrollment/session).
func rawPost(ctx context.Context, hc *http.Client, base *url.URL, path, msgType string, payload, out any) error {
	body, err := envelope(msgType, "", "", payload)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base.JoinPath(path).String(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(protocol.HeaderProtocol, strconv.Itoa(protocol.Version))
	resp, err := hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return decodeError(resp)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(out)
}

// EnrollResult is returned by Enroll for display before saving.
type EnrollResult struct {
	Identity              *identity.Identity
	ControllerFingerprint string
}

// Enroll performs the two-step enrollment with a fresh Ed25519 key. If
// pinnedKey is non-nil the controller must present that identity key.
func Enroll(ctx context.Context, controllerURL string, allowInsecure bool, code string, dev protocol.DeviceInfo, pinnedKey []byte) (*EnrollResult, error) {
	base, err := config.CheckControllerURL(controllerURL, allowInsecure)
	if err != nil {
		return nil, err
	}
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		return nil, err
	}
	if dev.AgentVersion == "" {
		dev.AgentVersion = versioning.Version
	}
	hc := &http.Client{}
	var begin protocol.EnrollBeginResponse
	if err := rawPost(ctx, hc, base, "/api/v1/enrollment/begin", protocol.MsgEnrollBegin,
		protocol.EnrollBeginRequest{Code: code, PublicKey: pub, Device: dev}, &begin); err != nil {
		return nil, err
	}
	if len(begin.ControllerKey) != ed25519.PublicKeySize ||
		!ed25519.Verify(begin.ControllerKey, protocol.EnrollChallengeMessage(begin.EnrollmentID, begin.Nonce, pub, begin.ProjectID), begin.ControllerSignature) {
		return nil, ErrControllerIdentity
	}
	if pinnedKey != nil && !bytes.Equal(pinnedKey, begin.ControllerKey) {
		return nil, ErrControllerIdentity
	}
	var done protocol.EnrollCompleteResponse
	if err := rawPost(ctx, hc, base, "/api/v1/enrollment/complete", protocol.MsgEnrollComplete,
		protocol.EnrollCompleteRequest{EnrollmentID: begin.EnrollmentID,
			Signature: ed25519.Sign(priv, protocol.EnrollProofMessage(begin.EnrollmentID, begin.Nonce, begin.ProjectID))}, &done); err != nil {
		return nil, err
	}
	id := &identity.Identity{ControllerURL: base.String(), ControllerKey: begin.ControllerKey, ProjectID: done.ProjectID,
		ProjectName: done.ProjectName, DeviceID: done.DeviceID, DeviceName: done.DeviceName, ContributorName: dev.ContributorName,
		PrivateSeed: priv.Seed(), EnrolledAt: time.Now().UTC()}
	return &EnrollResult{Identity: id, ControllerFingerprint: secure.Fingerprint(begin.ControllerKey)}, nil
}

// OpenSession obtains a fresh bearer token, verifying the controller's identity.
func (c *Client) OpenSession(ctx context.Context) (*protocol.SessionResponse, error) {
	c.refreshMu.Lock()
	defer c.refreshMu.Unlock()
	return c.openSession(ctx)
}

func (c *Client) openSession(ctx context.Context) (*protocol.SessionResponse, error) {
	nonce := secure.RandomBytes(24)
	ts := time.Now().UnixMilli()
	var resp protocol.SessionResponse
	err := rawPost(ctx, c.hc, c.base, "/api/v1/worker/session", protocol.MsgSession, protocol.SessionRequest{
		DeviceID: c.id.DeviceID, TimestampMS: ts, Nonce: nonce,
		Signature: ed25519.Sign(c.id.Key(), protocol.SessionMessage(c.id.DeviceID, ts, nonce))}, &resp)
	if err != nil {
		return nil, err
	}
	if !ed25519.Verify(c.id.ControllerKey, protocol.SessionAckMessage(c.id.DeviceID, nonce, resp.SessionID), resp.ControllerSignature) {
		return nil, ErrControllerIdentity
	}
	c.mu.Lock()
	c.token, c.expires, c.heartbeat = resp.Token, resp.ExpiresAt, resp.HeartbeatIntervalSec
	c.mu.Unlock()
	return &resp, nil
}

// ensureSession refreshes the token when missing or close to expiry.
func (c *Client) ensureSession(ctx context.Context) (string, error) {
	return c.refresh(ctx, "")
}

// refresh returns a usable token. If stale is non-empty the caller saw it
// rejected (401); a new session is opened only if no other goroutine has
// already replaced that token, so concurrent 401s cause one refresh, not a
// cascade of sessions revoking each other.
func (c *Client) refresh(ctx context.Context, stale string) (string, error) {
	valid := func() (string, bool) {
		c.mu.Lock()
		defer c.mu.Unlock()
		ok := c.token != "" && c.token != stale && time.Until(c.expires) > 2*time.Minute
		return c.token, ok
	}
	if tok, ok := valid(); ok {
		return tok, nil
	}
	c.refreshMu.Lock()
	defer c.refreshMu.Unlock()
	if tok, ok := valid(); ok {
		return tok, nil
	}
	if _, err := c.openSession(ctx); err != nil {
		return "", err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.token, nil
}

// InvalidateSession forces a new session on the next request.
func (c *Client) InvalidateSession() {
	c.mu.Lock()
	c.token = ""
	c.mu.Unlock()
}

// Do sends an authenticated request. body may be nil. A 401 triggers one
// session refresh and retry.
func (c *Client) Do(ctx context.Context, method, path string, query url.Values, body []byte, contentType string, out any) (int, error) {
	for attempt := 0; attempt < 2; attempt++ {
		tok, err := c.ensureSession(ctx)
		if err != nil {
			return 0, err
		}
		u := c.base.JoinPath(path)
		if query != nil {
			u.RawQuery = query.Encode()
		}
		var rd io.Reader
		if body != nil {
			rd = bytes.NewReader(body)
			c.tx.Add(int64(len(body)))
		}
		req, err := http.NewRequestWithContext(ctx, method, u.String(), rd)
		if err != nil {
			return 0, err
		}
		req.Header.Set("Authorization", "Bearer "+tok)
		req.Header.Set(protocol.HeaderProtocol, strconv.Itoa(protocol.Version))
		if contentType != "" {
			req.Header.Set("Content-Type", contentType)
		}
		resp, err := c.hc.Do(req)
		if err != nil {
			return 0, err
		}
		if resp.StatusCode == http.StatusUnauthorized && attempt == 0 {
			io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
			resp.Body.Close()
			if _, err := c.refresh(ctx, tok); err != nil {
				return 0, err
			}
			continue
		}
		code := resp.StatusCode
		if code >= 400 {
			err := decodeError(resp)
			resp.Body.Close()
			return code, err
		}
		if out != nil && code != http.StatusNoContent {
			cr := &countingReader{r: io.LimitReader(resp.Body, 8<<20), n: &c.rx}
			err = json.NewDecoder(cr).Decode(out)
		}
		resp.Body.Close()
		return code, err
	}
	return 0, errors.New("unauthorized after session refresh")
}

// Send posts an enveloped message and decodes the JSON response.
func (c *Client) Send(ctx context.Context, path, msgType string, payload, out any) (int, error) {
	body, err := envelope(msgType, c.id.ProjectID, c.id.DeviceID, payload)
	if err != nil {
		return 0, err
	}
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	return c.Do(ctx, http.MethodPost, path, nil, body, "application/json", out)
}

// Heartbeat sends a heartbeat.
func (c *Client) Heartbeat(ctx context.Context, req protocol.HeartbeatRequest) (*protocol.HeartbeatResponse, error) {
	var resp protocol.HeartbeatResponse
	_, err := c.Send(ctx, "/api/v1/worker/heartbeat", protocol.MsgHeartbeat, req, &resp)
	return &resp, err
}

// NextJob long-polls for a job; nil means none available.
func (c *Client) NextJob(ctx context.Context, wait time.Duration) (*protocol.JobAssignment, error) {
	var a protocol.JobAssignment
	ctx, cancel := context.WithTimeout(ctx, wait+30*time.Second)
	defer cancel()
	code, err := c.Do(ctx, http.MethodGet, "/api/v1/worker/jobs/next", url.Values{"wait": {strconv.Itoa(int(wait.Seconds()))}}, nil, "", &a)
	if err != nil {
		return nil, err
	}
	if code == http.StatusNoContent {
		return nil, nil
	}
	return &a, nil
}

// Me returns the worker's own record.
func (c *Client) Me(ctx context.Context) (*protocol.WorkerSelf, error) {
	var me protocol.WorkerSelf
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	_, err := c.Do(ctx, http.MethodGet, "/api/v1/worker/me", nil, nil, "", &me)
	return &me, err
}

// Commands fetches pending commands.
func (c *Client) Commands(ctx context.Context) ([]protocol.Command, error) {
	var out struct {
		Commands []protocol.Command `json:"commands"`
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	_, err := c.Do(ctx, http.MethodGet, "/api/v1/worker/commands", nil, nil, "", &out)
	return out.Commands, err
}

// AckCommand advances a command's state.
func (c *Client) AckCommand(ctx context.Context, id, state string, result any) error {
	var raw json.RawMessage
	if result != nil {
		raw, _ = json.Marshal(result)
	}
	_, err := c.Send(ctx, "/api/v1/worker/commands/"+url.PathEscape(id)+"/ack", protocol.MsgCommandAck,
		protocol.CommandAckRequest{State: state, Result: raw}, nil)
	return err
}

// Job lifecycle calls.

// Ack confirms an assignment.
func (c *Client) Ack(ctx context.Context, a *protocol.JobAssignment) error {
	_, err := c.Send(ctx, "/api/v1/worker/jobs/"+a.AttemptID+"/ack", protocol.MsgJobAck, protocol.JobAckRequest{FencingToken: a.FencingToken}, nil)
	return err
}

// Progress reports progress.
func (c *Client) Progress(ctx context.Context, attemptID string, p protocol.JobProgressRequest) error {
	_, err := c.Send(ctx, "/api/v1/worker/jobs/"+attemptID+"/progress", protocol.MsgJobProgress, p, nil)
	return err
}

// Complete submits a result.
func (c *Client) Complete(ctx context.Context, attemptID string, req protocol.JobCompleteRequest) (*protocol.JobCompleteResponse, error) {
	var resp protocol.JobCompleteResponse
	_, err := c.Send(ctx, "/api/v1/worker/jobs/"+attemptID+"/complete", protocol.MsgJobComplete, req, &resp)
	return &resp, err
}

// Fail reports a failure.
func (c *Client) Fail(ctx context.Context, attemptID string, req protocol.JobFailRequest) error {
	_, err := c.Send(ctx, "/api/v1/worker/jobs/"+attemptID+"/fail", protocol.MsgJobFail, req, nil)
	return err
}

// ShipLogs sends a batch of log lines.
func (c *Client) ShipLogs(ctx context.Context, lines []protocol.LogLine) error {
	_, err := c.Send(ctx, "/api/v1/worker/logs", protocol.MsgLogBatch, protocol.LogBatch{Lines: lines}, nil)
	return err
}

type countingReader struct {
	r io.Reader
	n *atomic.Int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n.Add(int64(n))
	return n, err
}

// throttle sleeps to keep throughput under Limit bytes/s.
func (c *Client) throttle(start time.Time, sent int64) {
	lim := c.Limit.Load()
	if lim <= 0 {
		return
	}
	want := time.Duration(float64(sent) / float64(lim) * float64(time.Second))
	if el := time.Since(start); el < want {
		time.Sleep(want - el)
	}
}

// Upload sends a file as a resumable artifact upload and returns its id.
func (c *Client) Upload(ctx context.Context, attemptID string, fencing int64, path, kind, contentType string, progress func(sent, total int64)) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	size, err := io.Copy(h, f)
	if err != nil {
		return "", err
	}
	sum := hex.EncodeToString(h.Sum(nil))
	var st protocol.UploadStatus
	if _, err := c.Send(ctx, "/api/v1/artifacts/uploads", protocol.MsgUploadBegin,
		protocol.UploadBeginRequest{Kind: kind, Size: size, SHA256: sum, ContentType: contentType, AttemptID: attemptID, FencingToken: fencing}, &st); err != nil {
		return "", err
	}
	if st.State == "complete" {
		return st.ArtifactID, nil
	}
	chunk := st.ChunkSize
	if chunk <= 0 || chunk > 8<<20 {
		chunk = 8 << 20
	}
	buf := make([]byte, chunk)
	start := time.Now()
	var sentTotal int64
	failures := 0
	for st.Offset < size {
		n, err := f.ReadAt(buf[:min(chunk, size-st.Offset)], st.Offset)
		if err != nil && !errors.Is(err, io.EOF) {
			return "", err
		}
		var next protocol.UploadStatus
		cctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
		_, err = c.Do(cctx, http.MethodPut, "/api/v1/artifacts/uploads/"+st.UploadID, url.Values{"offset": {strconv.FormatInt(st.Offset, 10)}},
			buf[:n], "application/octet-stream", &next)
		cancel()
		if err != nil {
			if protocol.IsCode(err, protocol.ErrOffsetMismatch) || isTransient(err) {
				failures++
				if failures > 20 {
					return "", fmt.Errorf("upload failed repeatedly: %w", err)
				}
				if sleepCtx(ctx, backoff(failures)) != nil {
					return "", ctx.Err()
				}
				// Resume from the controller's authoritative offset.
				var cur protocol.UploadStatus
				if _, serr := c.Do(ctx, http.MethodGet, "/api/v1/artifacts/uploads/"+st.UploadID, nil, nil, "", &cur); serr == nil {
					st.Offset = cur.Offset
				}
				continue
			}
			return "", err
		}
		failures = 0
		sentTotal += int64(n)
		st.Offset = next.Offset
		if progress != nil {
			progress(st.Offset, size)
		}
		c.throttle(start, sentTotal)
	}
	var done protocol.UploadStatus
	if _, err := c.Do(ctx, http.MethodPost, "/api/v1/artifacts/uploads/"+st.UploadID+"/complete", nil, []byte(`{}`), "application/json", &done); err != nil {
		return "", err
	}
	return done.ArtifactID, nil
}

// Download fetches an artifact into dst with Range resume, verifying sha256.
func (c *Client) Download(ctx context.Context, ref protocol.ArtifactRef, dst string) error {
	if st, err := os.Stat(dst); err == nil && st.Size() == ref.Size {
		if ok, _ := fileHasSHA(dst, ref.SHA256); ok {
			return nil // cache hit
		}
	}
	part := dst + ".part"
	failures := 0
	for {
		err := c.downloadOnce(ctx, ref, part)
		if err == nil {
			break
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		failures++
		if failures > 10 || !isTransient(err) {
			return err
		}
		if sleepCtx(ctx, backoff(failures)) != nil {
			return ctx.Err()
		}
	}
	ok, err := fileHasSHA(part, ref.SHA256)
	if err != nil {
		return err
	}
	if !ok {
		os.Remove(part)
		return fmt.Errorf("artifact %s failed sha256 verification", ref.ID)
	}
	return os.Rename(part, dst)
}

func (c *Client) downloadOnce(ctx context.Context, ref protocol.ArtifactRef, part string) error {
	tok, err := c.ensureSession(ctx)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(part, os.O_CREATE|os.O_WRONLY, 0o640)
	if err != nil {
		return err
	}
	defer f.Close()
	have, _ := f.Seek(0, io.SeekEnd)
	if have >= ref.Size {
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base.JoinPath("/api/v1/artifacts/"+ref.ID).String(), nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	if have > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", have))
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusPartialContent:
	case http.StatusOK:
		if have > 0 { // server ignored Range: restart
			if err := f.Truncate(0); err != nil {
				return err
			}
			if _, err := f.Seek(0, io.SeekStart); err != nil {
				return err
			}
		}
	case http.StatusUnauthorized:
		_, _ = c.refresh(ctx, tok)
		return &protocol.APIError{Status: 401, Code: protocol.ErrUnauthorized, Message: "session expired"}
	default:
		return decodeError(resp)
	}
	start := time.Now()
	buf := make([]byte, 256<<10)
	var got int64
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := f.Write(buf[:n]); err != nil {
				return err
			}
			got += int64(n)
			c.rx.Add(int64(n))
			c.throttle(start, got)
		}
		if rerr == io.EOF {
			return nil
		}
		if rerr != nil {
			return rerr
		}
	}
}

func fileHasSHA(path, want string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, err
	}
	return hex.EncodeToString(h.Sum(nil)) == want, nil
}

// isTransient reports errors worth retrying (network failures, 5xx, 429, 401).
func isTransient(err error) bool {
	var ae *protocol.APIError
	if errors.As(err, &ae) {
		return ae.Status >= 500 || ae.Status == 429 || ae.Status == 401 || ae.Code == protocol.ErrOffsetMismatch
	}
	return true
}

// IsTransient exposes the retry classification.
func IsTransient(err error) bool { return isTransient(err) }

func backoff(n int) time.Duration {
	d := time.Duration(1<<min(n, 6)) * time.Second
	return min(d, 60*time.Second)
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
