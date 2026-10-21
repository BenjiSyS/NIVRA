package enrollment

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/sha256"
	"errors"
	"strings"

	"nivra/internal/shared/secure"
)

// Project codes use Crockford base32 (no I, L, O, U) so they survive being
// read aloud or retyped. 16 symbols = 80 bits of entropy; combined with
// per-IP rate limiting, online guessing is infeasible.
const (
	crockford   = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
	codeSymbols = 16
	codePrefix  = "NV"
)

// ErrMalformedCode means the input cannot be a project code.
var ErrMalformedCode = errors.New("malformed project code")

// GenerateCode returns a fresh code formatted as NV-XXXX-XXXX-XXXX-XXXX.
func GenerateCode() string {
	raw := secure.RandomBytes(codeSymbols)
	var b strings.Builder
	b.WriteString(codePrefix)
	for i, r := range raw {
		if i%4 == 0 {
			b.WriteByte('-')
		}
		// 256 is divisible by 32, so r%32 is unbiased.
		b.WriteByte(crockford[int(r)%32])
	}
	return b.String()
}

// NormalizeCode canonicalises user input: case-insensitive, tolerant of
// spaces/dashes and of the common confusions O→0, I/L→1.
func NormalizeCode(in string) (string, error) {
	s := strings.ToUpper(strings.TrimSpace(in))
	s = strings.NewReplacer("-", "", " ", "", "_", "").Replace(s)
	if strings.HasPrefix(s, codePrefix) && len(s) == codeSymbols+len(codePrefix) {
		s = s[len(codePrefix):]
	}
	if len(s) != codeSymbols {
		return "", ErrMalformedCode
	}
	var b strings.Builder
	for _, c := range s {
		switch c {
		case 'O':
			c = '0'
		case 'I', 'L':
			c = '1'
		}
		if !strings.ContainsRune(crockford, c) {
			return "", ErrMalformedCode
		}
		b.WriteRune(c)
	}
	return b.String(), nil
}

// FormatCode renders a normalized code for display.
func FormatCode(norm string) string {
	var b strings.Builder
	b.WriteString(codePrefix)
	for i := 0; i < len(norm); i += 4 {
		b.WriteByte('-')
		b.WriteString(norm[i:min(i+4, len(norm))])
	}
	return b.String()
}

// HashCode returns the keyed lookup hash of a normalized code.
func HashCode(key []byte, norm string) []byte {
	m := hmac.New(sha256.New, key)
	m.Write([]byte("nivra-project-code-v1|"))
	m.Write([]byte(norm))
	return m.Sum(nil)
}

// EncryptCode seals the normalized code with AES-256-GCM, bound to the project id.
func EncryptCode(key []byte, projectID, norm string) ([]byte, error) {
	gcm, err := newGCM(key)
	if err != nil {
		return nil, err
	}
	nonce := secure.RandomBytes(gcm.NonceSize())
	return gcm.Seal(nonce, nonce, []byte(norm), []byte("nivra-code|"+projectID)), nil
}

// DecryptCode reverses EncryptCode.
func DecryptCode(key []byte, projectID string, sealed []byte) (string, error) {
	gcm, err := newGCM(key)
	if err != nil {
		return "", err
	}
	if len(sealed) < gcm.NonceSize() {
		return "", errors.New("ciphertext too short")
	}
	pt, err := gcm.Open(nil, sealed[:gcm.NonceSize()], sealed[gcm.NonceSize():], []byte("nivra-code|"+projectID))
	if err != nil {
		return "", errors.New("project code ciphertext failed authentication")
	}
	return string(pt), nil
}

func newGCM(key []byte) (cipher.AEAD, error) {
	blk, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(blk)
}
