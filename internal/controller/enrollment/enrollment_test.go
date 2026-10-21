package enrollment_test

import (
	"context"
	"crypto/ed25519"
	"errors"
	"strings"
	"testing"
	"time"

	"nivra/internal/controller/core"
	"nivra/internal/controller/enrollment"
	"nivra/internal/controller/projects"
	"nivra/internal/shared/protocol"
	"nivra/internal/testutil"
)

func TestCodeFormatAndNormalization(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 2000; i++ {
		c := enrollment.GenerateCode()
		if len(c) != len("NV-XXXX-XXXX-XXXX-XXXX") || !strings.HasPrefix(c, "NV-") {
			t.Fatalf("bad format %q", c)
		}
		if seen[c] {
			t.Fatalf("duplicate code %q", c)
		}
		seen[c] = true
		n, err := enrollment.NormalizeCode(strings.ToLower(strings.ReplaceAll(c, "-", " ")))
		if err != nil || enrollment.FormatCode(n) != c {
			t.Fatalf("roundtrip %q -> %q %v", c, n, err)
		}
	}
	if n, _ := enrollment.NormalizeCode("NV-OOOO-IIII-LLLL-0000"); n != "0000111111110000" {
		t.Fatalf("confusable mapping failed: %q", n)
	}
	for _, bad := range []string{"", "NV-1234", "NV-UUUU-UUUU-UUUU-UUUU", "abc"} {
		if _, err := enrollment.NormalizeCode(bad); err == nil {
			t.Fatalf("accepted %q", bad)
		}
	}
}

func TestCodeEncryptionBoundToProject(t *testing.T) {
	key := make([]byte, 32)
	sealed, err := enrollment.EncryptCode(key, "project-a", "ABCDEFGHJKMNPQRS")
	if err != nil {
		t.Fatal(err)
	}
	if got, err := enrollment.DecryptCode(key, "project-a", sealed); err != nil || got != "ABCDEFGHJKMNPQRS" {
		t.Fatalf("decrypt: %q %v", got, err)
	}
	if _, err := enrollment.DecryptCode(key, "project-b", sealed); err == nil {
		t.Fatal("ciphertext accepted for another project")
	}
}

type dev struct {
	id   string
	priv ed25519.PrivateKey
}

func enroll(t *testing.T, en *enrollment.Service, code string) (*dev, error) {
	t.Helper()
	ctx := context.Background()
	pub, priv, _ := ed25519.GenerateKey(nil)
	b, err := en.Begin(ctx, protocol.EnrollBeginRequest{Code: code, PublicKey: pub, Device: protocol.DeviceInfo{Name: "d"}}, "test")
	if err != nil {
		return nil, err
	}
	if !ed25519.Verify(b.ControllerKey, protocol.EnrollChallengeMessage(b.EnrollmentID, b.Nonce, pub, b.ProjectID), b.ControllerSignature) {
		t.Fatal("controller signature on challenge invalid")
	}
	c, err := en.Complete(ctx, protocol.EnrollCompleteRequest{EnrollmentID: b.EnrollmentID,
		Signature: ed25519.Sign(priv, protocol.EnrollProofMessage(b.EnrollmentID, b.Nonce, b.ProjectID))}, "test")
	if err != nil {
		return nil, err
	}
	return &dev{id: c.DeviceID, priv: priv}, nil
}

func session(en *enrollment.Service, d *dev, nonce []byte, ts time.Time) (protocol.SessionResponse, error) {
	ms := ts.UnixMilli()
	return en.OpenSession(context.Background(), protocol.SessionRequest{DeviceID: d.id, TimestampMS: ms, Nonce: nonce,
		Signature: ed25519.Sign(d.priv, protocol.SessionMessage(d.id, ms, nonce))}, "test")
}

func TestEnrollmentSessionsReplayRotationRevocation(t *testing.T) {
	d := testutil.Deps(t)
	ctx := context.Background()
	ps := projects.New(d)
	p, ci, err := ps.Create(ctx, core.System, "P")
	if err != nil {
		t.Fatal(err)
	}
	en := enrollment.New(d)

	if _, err := enroll(t, en, "NV-0000-0000-0000-0000"); !errors.Is(err, enrollment.ErrInvalidCode) {
		t.Fatalf("wrong code: %v", err)
	}
	dv, err := enroll(t, en, ci.Code)
	if err != nil {
		t.Fatal(err)
	}

	// Session issuance, server signature, authentication.
	nonce := []byte("0123456789abcdef0123")
	s, err := session(en, dv, nonce, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if !ed25519.Verify(d.Secrets.PublicKey(), protocol.SessionAckMessage(dv.id, nonce, s.SessionID), s.ControllerSignature) {
		t.Fatal("session ack signature invalid")
	}
	if id, err := en.Authenticate(ctx, s.Token); err != nil || id.DeviceID != dv.id {
		t.Fatalf("authenticate: %v", err)
	}
	// Replay of the identical signed request is refused.
	if _, err := session(en, dv, nonce, time.UnixMilli(0).Add(time.Duration(time.Now().UnixMilli())*time.Millisecond)); !errors.Is(err, enrollment.ErrReplay) {
		t.Fatalf("replay: want ErrReplay, got %v", err)
	}
	// Clock skew is refused.
	if _, err := session(en, dv, []byte("another-nonce-123456"), time.Now().Add(-10*time.Minute)); !errors.Is(err, enrollment.ErrClockSkew) {
		t.Fatalf("skew: %v", err)
	}
	// Wrong key cannot open a session for the device.
	_, other, _ := ed25519.GenerateKey(nil)
	if _, err := session(en, &dev{id: dv.id, priv: other}, []byte("third-nonce-12345678"), time.Now()); !errors.Is(err, enrollment.ErrBadSignature) {
		t.Fatalf("forged signature: %v", err)
	}
	// A new session revokes the previous one.
	s2, err := session(en, dv, []byte("fourth-nonce-1234567"), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := en.Authenticate(ctx, s.Token); err == nil {
		t.Fatal("old session still valid after a new one was opened")
	}

	// Rotation: old code stops enrolling, enrolled device keeps working.
	nc, err := ps.Rotate(ctx, core.System, p.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := enroll(t, en, ci.Code); !errors.Is(err, enrollment.ErrInvalidCode) {
		t.Fatalf("old code after rotation: %v", err)
	}
	if _, err := enroll(t, en, nc.Code); err != nil {
		t.Fatalf("new code: %v", err)
	}
	if _, err := en.Authenticate(ctx, s2.Token); err != nil {
		t.Fatalf("enrolled device broken by rotation: %v", err)
	}

	// Revocation: sessions fail with ErrDeviceRevoked and the key cannot re-enroll.
	if _, err := d.DB.Exec(ctx, `UPDATE devices SET status='revoked' WHERE id=$1`, dv.id); err != nil {
		t.Fatal(err)
	}
	if _, err := en.Authenticate(ctx, s2.Token); !errors.Is(err, enrollment.ErrDeviceRevoked) {
		t.Fatalf("revoked auth: %v", err)
	}
	if _, err := session(en, dv, []byte("fifth-nonce-12345678"), time.Now()); !errors.Is(err, enrollment.ErrDeviceRevoked) {
		t.Fatalf("revoked session: %v", err)
	}
}
