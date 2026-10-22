package main

import (
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func freeAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().String()
}

func TestChooseUIAddrFree(t *testing.T) {
	a := freeAddr(t)
	use, existing, err := chooseUIAddr(a)
	if err != nil || use != a || existing != "" {
		t.Fatalf("got %q %q %v", use, existing, err)
	}
}

func TestChooseUIAddrOpensRunningNivra(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/local/api/status" {
			_, _ = w.Write([]byte(`{"state":"idle","version":"1.0","hardware":{}}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()
	addr := strings.TrimPrefix(srv.URL, "http://")
	use, existing, err := chooseUIAddr(addr)
	if err != nil || existing != "http://"+addr+"/app" || use != addr {
		t.Fatalf("got %q %q %v", use, existing, err)
	}
}

func TestChooseUIAddrPicksAnotherPortWhenOtherProgramHoldsIt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("<html>some other web server</html>"))
	}))
	defer srv.Close()
	addr := strings.TrimPrefix(srv.URL, "http://")
	use, existing, err := chooseUIAddr(addr)
	if err != nil || existing != "" || use == addr || !strings.HasPrefix(use, "127.0.0.1:") {
		t.Fatalf("got %q %q %v", use, existing, err)
	}
}

func TestChooseUIAddrPortHeldByNonHTTP(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	use, existing, err := chooseUIAddr(ln.Addr().String())
	if err != nil || existing != "" || use == ln.Addr().String() {
		t.Fatalf("got %q %q %v", use, existing, err)
	}
}

func TestOneLine(t *testing.T) {
	if got := oneLine("first\nsecond"); got != "first" {
		t.Fatal(got)
	}
	if got := oneLine(strings.Repeat("x", 400)); len(got) != 303 {
		t.Fatal(len(got))
	}
}
