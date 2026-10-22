package localui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"nivra/internal/shared/config"
)

// Sign-in with a username and password: the app signs in once, asks the Nivra
// server for a personal access key (API token), keeps ONLY that key, and signs the
// session out again. The password is never stored or logged.

// mintToken signs in at base, creates a personal API token named for this computer
// and signs out. The session cookie is handled by hand (no cookie jar) so it works
// for the controller's __Host- Secure cookie as well as plain-HTTP test servers.
func (a *App) mintToken(ctx context.Context, base, username, password string) (string, error) {
	hc := &http.Client{Timeout: 30 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}

	do := func(method, path string, body any, cookie *http.Cookie, csrf string, out any) (*http.Response, error) {
		var rd io.Reader
		if body != nil {
			b, _ := json.Marshal(body)
			rd = bytes.NewReader(b)
		}
		req, err := http.NewRequestWithContext(ctx, method, base+"/api/v1"+path, rd)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		if cookie != nil {
			req.AddCookie(cookie)
		}
		if csrf != "" {
			req.Header.Set("X-Nivra-CSRF", csrf)
		}
		resp, err := hc.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if resp.StatusCode >= 300 {
			var e struct {
				Error struct{ Message string } `json:"error"`
			}
			_ = json.Unmarshal(b, &e)
			if e.Error.Message == "" {
				e.Error.Message = http.StatusText(resp.StatusCode)
			}
			return resp, &ctlErr{resp.StatusCode, e.Error.Message}
		}
		if out != nil {
			if err := json.Unmarshal(b, out); err != nil {
				return resp, err
			}
		}
		return resp, nil
	}

	var login struct {
		CSRF string `json:"csrf_token"`
	}
	resp, err := do("POST", "/auth/login", map[string]string{"username": username, "password": password}, nil, "", &login)
	if err != nil {
		return "", err
	}
	var session *http.Cookie
	for _, c := range resp.Cookies() {
		if c.HttpOnly && c.Value != "" {
			session = c
		}
	}
	if session == nil || login.CSRF == "" {
		return "", errors.New("the Nivra server did not start a session; is the address right?")
	}
	// Always sign the session out again, even if the token request fails.
	defer func() {
		lctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(lctx, "POST", base+"/api/v1/auth/logout", nil)
		if err != nil {
			return
		}
		req.AddCookie(session)
		req.Header.Set("X-Nivra-CSRF", login.CSRF)
		if r, err := hc.Do(req); err == nil {
			r.Body.Close()
		}
	}()

	host, _ := os.Hostname()
	if host == "" {
		host = "this computer"
	}
	var created struct {
		Token string `json:"token"`
	}
	if _, err := do("POST", "/tokens", map[string]any{"name": "Nivra App on " + host, "days": 365}, session, login.CSRF, &created); err != nil {
		return "", err
	}
	if created.Token == "" {
		return "", errors.New("the Nivra server did not return an access key")
	}
	return created.Token, nil
}

func (a *App) registerLogin(mux *http.ServeMux) {
	mux.HandleFunc("POST /local/api/app/login", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ControllerURL string `json:"controller_url"`
			Username      string `json:"username"`
			Password      string `json:"password"`
			Project       string `json:"project"`
		}
		if !decodeBody(w, r, &req) {
			return
		}
		u, err := config.CheckControllerURL(req.ControllerURL, a.AllowInsecure)
		if err != nil {
			writeErr(w, 400, "That Nivra server address does not look right. It should start with https://")
			return
		}
		username := strings.TrimSpace(req.Username)
		if username == "" || req.Password == "" {
			writeErr(w, 400, "Enter your username and password.")
			return
		}
		base := strings.TrimRight(u.String(), "/")
		ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
		defer cancel()
		token, err := a.mintToken(ctx, base, username, req.Password)
		if err != nil {
			writeErr(w, 400, friendlyErr(err))
			return
		}
		project := strings.TrimSpace(req.Project)
		if _, _, err := a.verify(ctx, base, token, project); err != nil {
			writeErr(w, 400, err.Error())
			return
		}
		a.mu.Lock()
		a.cfg.ControllerURL, a.cfg.Token, a.cfg.Project = base, token, project
		err = a.saveCfgLocked()
		a.mu.Unlock()
		if err != nil {
			writeErr(w, 500, "could not save settings: "+err.Error())
			return
		}
		a.refresh(r.Context())
		a.poke()
		writeJSON(w, 200, a.configView())
	})
}
