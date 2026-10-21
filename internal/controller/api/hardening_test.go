package api

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"nivra/internal/controller/core"
	"nivra/internal/shared/config"
)

func TestRateKeyGroupsIPv6Prefixes(t *testing.T) {
	if rateKey("203.0.113.7") != "203.0.113.7" {
		t.Fatal("IPv4 must be used as-is")
	}
	a, b := rateKey("2001:db8:1:2::1"), rateKey("2001:db8:1:2:ffff::9")
	if a != b || a != "2001:db8:1:2::/64" {
		t.Fatalf("IPv6 /64 grouping failed: %q %q", a, b)
	}
	if rateKey("2001:db8:1:3::1") == a {
		t.Fatal("different /64 grouped together")
	}
}

func TestClientIPIgnoresForwardedFor(t *testing.T) {
	s := &Server{d: &core.Deps{Cfg: config.Controller{TrustProxyHeaders: true}}}
	r := httptest.NewRequest("GET", "/", nil)
	r.RemoteAddr = "127.0.0.1:5555"
	r.Header.Set("X-Forwarded-For", "198.51.100.1")
	if got := s.clientIP(r); got != "127.0.0.1" {
		t.Fatalf("X-Forwarded-For must not be trusted, got %s", got)
	}
	r.Header.Set("CF-Connecting-IP", "198.51.100.2")
	if got := s.clientIP(r); got != "198.51.100.2" {
		t.Fatalf("CF-Connecting-IP from loopback: got %s", got)
	}
	r.RemoteAddr = "192.168.1.9:4444"
	if got := s.clientIP(r); got != "192.168.1.9" {
		t.Fatalf("headers from non-loopback peers must be ignored, got %s", got)
	}
}

func TestHostSplit(t *testing.T) {
	s := &Server{d: &core.Deps{Cfg: config.Controller{APIHosts: []string{"api.example.com"}, StudioHosts: []string{"studio.example.com"}}}}
	ok := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) })
	h := s.hostSplit(ok)
	cases := []struct {
		host, path string
		want       int
	}{
		{"api.example.com", "/api/v1/worker/heartbeat", 200},
		{"api.example.com", "/api/v1/enrollment/begin", 200},
		{"api.example.com", "/api/v1/health", 200},
		{"api.example.com", "/api/v1/auth/login", 404},
		{"api.example.com", "/api/v1/projects", 404},
		{"api.example.com", "/", 404},
		{"studio.example.com:443", "/api/v1/auth/login", 200},
		{"studio.example.com", "/", 200},
		{"studio.example.com", "/api/v1/worker/heartbeat", 404},
		{"localhost:8740", "/api/v1/auth/login", 200},
	}
	for _, c := range cases {
		r := httptest.NewRequest("GET", c.path, nil)
		r.Host = c.host
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != c.want {
			t.Errorf("%s%s: got %d want %d", c.host, c.path, w.Code, c.want)
		}
	}
}

func TestHTTPSPublicURLRequiresSecureSettings(t *testing.T) {
	c := config.DefaultController()
	c.PublicURL = "https://api.example.com"
	if c.Validate() == nil {
		t.Fatal("https public_url accepted without cookie_secure/trust_proxy_headers")
	}
	c.CookieSecure, c.TrustProxyHeaders = true, true
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestSecureCookieUsesHostPrefix(t *testing.T) {
	s := &Server{d: &core.Deps{Cfg: config.Controller{CookieSecure: true}}}
	w := httptest.NewRecorder()
	s.setSessionCookie(w, "tok", testTime())
	c := w.Result().Cookies()[0]
	if c.Name != "__Host-nivra_session" || !c.Secure || !c.HttpOnly || c.Path != "/" || c.Domain != "" {
		t.Fatalf("unexpected cookie %+v", c)
	}
}

func testTime() time.Time { return time.Now().Add(time.Hour) }
