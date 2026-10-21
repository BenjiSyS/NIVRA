package integration

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"

	"nivra/internal/controller/app"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/shared/config"
	"nivra/internal/testutil"
)

// TestAPITokens: tokens authenticate without cookies or CSRF, respect roles,
// and stop working when revoked.
func TestAPITokens(t *testing.T) {
	pool := testutil.DB(t)
	pool.Close()
	cfg := config.DefaultController()
	cfg.DataDir = t.TempDir()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.DatabaseURL = os.Getenv("NIVRA_TEST_DATABASE_URL")
	var opTok, viewTok, pid string
	c := startController(t, cfg, func(rt *app.Runtime) {
		ctx := context.Background()
		for _, u := range []struct{ name, role string }{{"operator1", auth.RoleOperator}, {"viewer", auth.RoleViewer}} {
			if _, err := rt.Auth.CreateUser(ctx, u.name, "", "token test password", u.role); err != nil {
				t.Fatal(err)
			}
		}
		var err error
		if opTok, err = rt.Auth.TokenForUsername(ctx, "operator1", "t", 0); err != nil {
			t.Fatal(err)
		}
		if viewTok, err = rt.Auth.TokenForUsername(ctx, "viewer", "t", 0); err != nil {
			t.Fatal(err)
		}
		p, _, _ := rt.Projects.Create(ctx, core.System, "Tok")
		pid = p.ID
	})
	defer c.stop(t)
	do := func(tok, method, path, body string) int {
		req, _ := http.NewRequest(method, c.url+path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		if tok != "" {
			req.Header.Set("Authorization", "Bearer "+tok)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	job := `{"job_type":"builtin.sleep","params":{"seconds":1}}`
	if code := do(opTok, "POST", "/api/v1/projects/"+pid+"/jobs", job); code != 201 {
		t.Fatalf("operator token POST without CSRF: %d", code)
	}
	if code := do(viewTok, "POST", "/api/v1/projects/"+pid+"/jobs", job); code != 403 {
		t.Fatalf("viewer token must be forbidden to create jobs: %d", code)
	}
	if code := do("nivra_pat_invalid", "GET", "/api/v1/projects", ""); code != 401 {
		t.Fatalf("bad token: %d", code)
	}
	// List and revoke the operator's own token through the API.
	req, _ := http.NewRequest("GET", c.url+"/api/v1/tokens", nil)
	req.Header.Set("Authorization", "Bearer "+opTok)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	var list struct {
		Tokens []struct{ ID string } `json:"tokens"`
	}
	json.NewDecoder(resp.Body).Decode(&list)
	resp.Body.Close()
	if len(list.Tokens) != 1 {
		t.Fatalf("token list: %+v", list)
	}
	if code := do(opTok, "DELETE", "/api/v1/tokens/"+list.Tokens[0].ID, ""); code != 200 {
		t.Fatalf("revoke: %d", code)
	}
	if code := do(opTok, "GET", "/api/v1/projects", ""); code != 401 {
		t.Fatalf("revoked token still works: %d", code)
	}
}
