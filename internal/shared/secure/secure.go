// Package secure holds small cryptographic helpers shared by controller and worker.
package secure

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
)

// RandomBytes returns n bytes from the OS CSPRNG. It panics if the CSPRNG
// fails, because continuing without randomness would be unsafe.
func RandomBytes(n int) []byte {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		panic("secure: crypto/rand failed: " + err.Error())
	}
	return b
}

// RandomToken returns a URL-safe random token with 256 bits of entropy.
func RandomToken() string { return base64.RawURLEncoding.EncodeToString(RandomBytes(32)) }

// HashToken returns SHA-256(token); bearer tokens are stored only as hashes.
func HashToken(token string) []byte {
	s := sha256.Sum256([]byte(token))
	return s[:]
}

// SHA256Hex returns the lowercase hex SHA-256 of b.
func SHA256Hex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// Fingerprint formats SHA-256(pub) as 8 groups of 4 hex digits; controller
// and worker both display identity keys this way so they can be compared.
func Fingerprint(pub []byte) string {
	h := SHA256Hex(pub)[:32]
	out := ""
	for i := 0; i < len(h); i += 4 {
		if i > 0 {
			out += ":"
		}
		out += h[i : i+4]
	}
	return out
}

// Equal compares two byte slices in constant time.
func Equal(a, b []byte) bool { return subtle.ConstantTimeCompare(a, b) == 1 }
