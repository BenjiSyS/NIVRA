package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
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

type roleEnv struct {
	url     string
	pid     string
	tokens  map[string]string // role label -> bearer token
	job1    string            // created by member1
	job2    string            // created by member2
	video1  string            // artifact of job1
	member2 string
	rt      *app.Runtime
}

func (r *roleEnv) call(tok, method, path string, body any) (int, []byte) {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, _ := http.NewRequest(method, r.url+strings.ReplaceAll(path, "{pid}", r.pid), rd)
	req.Header.Set("Content-Type", "application/json")
	if tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, b
}

func newRoleEnv(t *testing.T) (*roleEnv, *controller) {
	pool := testutil.DB(t)
	pool.Close()
	cfg := config.DefaultController()
	cfg.DataDir = t.TempDir()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.DatabaseURL = os.Getenv("NIVRA_TEST_DATABASE_URL")
	re := &roleEnv{tokens: map[string]string{}}
	ids := map[string]string{}
	c := startController(t, cfg, func(rt *app.Runtime) {
		ctx := context.Background()
		re.rt = rt
		for _, u := range []struct{ label, name, role string }{
			{"admin", "adm-one", auth.RoleAdmin}, {"operator", "op-one", auth.RoleOperator}, {"viewer", "view-one", auth.RoleViewer},
			{"member", "mem-one", auth.RoleMember}, {"member2", "mem-two", auth.RoleMember},
		} {
			u1, err := rt.Auth.CreateUser(ctx, u.name, "", "role matrix password", u.role)
			if err != nil {
				t.Fatal(err)
			}
			ids[u.label] = u1.ID
			if re.tokens[u.label], err = rt.Auth.TokenForUsername(ctx, u.name, "t", 0); err != nil {
				t.Fatal(err)
			}
		}
		p, _, _ := rt.Projects.Create(ctx, core.System, "Roles")
		re.pid = p.ID
		mkJob := func(owner string) string {
			var id string
			if err := rt.Deps.DB.QueryRow(ctx, `INSERT INTO jobs (project_id, job_type, category, status, created_by, params)
				VALUES ($1,'ai.generate','generation','ready',$2,'{}') RETURNING id::text`, p.ID, owner).Scan(&id); err != nil {
				t.Fatal(err)
			}
			return id
		}
		re.job1, re.job2 = mkJob(ids["member"]), mkJob(ids["member2"])
		re.member2 = ids["member2"]
		// A finished video for job1.
		var dev, att string
		if err := rt.Deps.DB.QueryRow(ctx, `INSERT INTO devices (project_id, public_key, name) VALUES ($1,$2,'d') RETURNING id::text`, p.ID, bytes.Repeat([]byte{7}, 32)).Scan(&dev); err != nil {
			t.Fatal(err)
		}
		if err := rt.Deps.DB.QueryRow(ctx, `INSERT INTO job_attempts (job_id, device_id, attempt_number, fencing_token, state, lease_expires_at)
			VALUES ($1,$2,1,nextval('fencing_seq'),'accepted', now() + interval '1 hour') RETURNING id::text`, re.job1, dev).Scan(&att); err != nil {
			t.Fatal(err)
		}
		info, err := rt.Artifacts.ImportBytes(ctx, p.ID, "video", append([]byte("\x00\x00\x00\x18ftypmp42"), make([]byte, 64)...), "")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := rt.Deps.DB.Exec(ctx, `UPDATE artifacts SET attempt_id=$2 WHERE id=$1`, info.ID, att); err != nil {
			t.Fatal(err)
		}
		re.video1 = info.ID
	})
	re.url = c.url
	return re, c
}

// TestRoleMatrix: every route against every role. 0 (reach) means "reachable" (any
// status other than 401/403, e.g. 400 for a deliberately empty body).
func TestRoleMatrix(t *testing.T) {
	re, c := newRoleEnv(t)
	defer c.stop(t)
	type row struct {
		method, path string
		body         any
		// expected status for admin, operator, viewer, member
		want [4]int
	}
	const reach = 0
	rows := []row{
		{"GET", "/api/v1/auth/me", nil, [4]int{200, 200, 200, 200}},
		{"GET", "/api/v1/projects", nil, [4]int{200, 200, 200, 200}},
		{"GET", "/api/v1/tokens", nil, [4]int{200, 200, 200, 200}},
		{"GET", "/api/v1/system", nil, [4]int{200, 200, 200, 403}},
		{"GET", "/api/v1/users", nil, [4]int{200, 403, 403, 403}},
		{"GET", "/api/v1/invites", nil, [4]int{200, 403, 403, 403}},
		{"POST", "/api/v1/invites", map[string]any{"role": "admin"}, [4]int{400, 403, 403, 403}},
		{"GET", "/api/v1/audit", nil, [4]int{200, 403, 403, 403}},
		{"GET", "/api/v1/projects/{pid}/overview", nil, [4]int{200, 200, 200, 403}},
		{"GET", "/api/v1/projects/{pid}/workers", nil, [4]int{200, 200, 200, 403}},
		{"GET", "/api/v1/projects/{pid}/jobs", nil, [4]int{200, 200, 200, 403}},
		{"POST", "/api/v1/projects/{pid}/jobs", map[string]any{"job_type": "builtin.sleep", "params": map[string]any{"seconds": 1}}, [4]int{201, 201, 403, 403}},
		{"GET", "/api/v1/projects/{pid}/logs", nil, [4]int{200, 200, 200, 403}},
		{"GET", "/api/v1/projects/{pid}/contributions", nil, [4]int{200, 200, 200, 403}},
		{"GET", "/api/v1/projects/{pid}/code", nil, [4]int{200, 200, 403, 403}},
		{"POST", "/api/v1/projects/{pid}/code/rotate", nil, [4]int{200, 403, 403, 403}},
		{"GET", "/api/v1/projects/{pid}/settings", nil, [4]int{200, 200, 403, 403}},
		{"PUT", "/api/v1/projects/{pid}/settings", map[string]any{"member_max_video_seconds": 600, "max_concurrent_long_videos": 2, "max_queued_jobs": 50, "intermediate_retention_hours": 24}, [4]int{200, 403, 403, 403}},
		{"GET", "/api/v1/projects/{pid}/limits", nil, [4]int{200, 200, 403, 200}},
		{"GET", "/api/v1/projects/{pid}/storage", nil, [4]int{200, 200, 403, 403}},
		{"POST", "/api/v1/projects/{pid}/storage/cleanup", map[string]any{"intermediates": true}, [4]int{200, 403, 403, 403}},
		{"GET", "/api/v1/projects/{pid}/models", nil, [4]int{200, 200, 200, 200}},
		{"GET", "/api/v1/projects/{pid}/checkpoints", nil, [4]int{200, 200, 200, 403}},
		{"GET", "/api/v1/projects/{pid}/artifacts", nil, [4]int{200, 200, 200, 403}},
		{"POST", "/api/v1/projects/{pid}/artifacts?kind=model_weights", nil, [4]int{reach, reach, 403, 403}},
		{"POST", "/api/v1/projects/{pid}/evaluate", map[string]any{"mode": "adherence"}, [4]int{reach, reach, 403, 403}},
		{"GET", "/api/v1/projects/{pid}/training/runs", nil, [4]int{200, 200, 200, 403}},
		{"POST", "/api/v1/projects/{pid}/training/runs", map[string]any{"name": "x"}, [4]int{reach, reach, 403, 403}},
		{"POST", "/api/v1/projects/{pid}/training/runs/00000000-0000-0000-0000-000000000000/stop", nil, [4]int{reach, reach, 403, 403}},
		{"GET", "/api/v1/projects/{pid}/datasets", nil, [4]int{200, 200, 200, 403}},
		{"POST", "/api/v1/projects/{pid}/datasets", map[string]any{}, [4]int{reach, reach, 403, 403}},
		{"POST", "/api/v1/projects/{pid}/datasets/shards", map[string]any{"jobs": 1}, [4]int{reach, reach, 403, 403}},
		{"POST", "/api/v1/projects/{pid}/models/import", map[string]any{}, [4]int{reach, reach, 403, 403}},
		{"POST", "/api/v1/projects/{pid}/generate", map[string]any{"prompt": "a cat"}, [4]int{reach, reach, 403, reach}},
		{"GET", "/api/v1/projects/{pid}/long-videos", nil, [4]int{200, 200, 200, 200}},
		{"GET", "/api/v1/projects/{pid}/generations", nil, [4]int{200, 200, 200, 200}},
		{"POST", "/api/v1/projects/{pid}/generations/00000000-0000-0000-0000-000000000000/cancel", nil, [4]int{404, 404, 403, 404}},
		{"POST", "/api/v1/projects/{pid}/generations/00000000-0000-0000-0000-000000000000/regenerate", nil, [4]int{404, 404, 403, 404}},
		{"DELETE", "/api/v1/projects/{pid}/generations/00000000-0000-0000-0000-000000000000", nil, [4]int{404, 404, 403, 404}},
		{"POST", "/api/v1/projects/{pid}/jobs/00000000-0000-0000-0000-000000000000/retry", nil, [4]int{reach, reach, 403, 403}},
	}
	for _, r := range rows {
		for i, label := range []string{"admin", "operator", "viewer", "member"} {
			code, body := re.call(re.tokens[label], r.method, r.path, r.body)
			want := r.want[i]
			ok := code == want || (want == reach && code != 401 && code != 403)
			if !ok {
				t.Errorf("%s %s as %s: HTTP %d (want %d) %s", r.method, r.path, label, code, want, strings.TrimSpace(string(body)))
			}
		}
	}
	// Unauthenticated callers get nothing.
	if code, _ := re.call("", "GET", "/api/v1/projects/{pid}/long-videos", nil); code != 401 {
		t.Errorf("anonymous: %d", code)
	}
}

func TestMemberSeesOnlyOwnWork(t *testing.T) {
	re, c := newRoleEnv(t)
	defer c.stop(t)
	m1, m2, op := re.tokens["member"], re.tokens["member2"], re.tokens["operator"]

	if code, _ := re.call(m1, "GET", "/api/v1/projects/{pid}/jobs/"+re.job1, nil); code != 200 {
		t.Fatalf("own job: %d", code)
	}
	if code, _ := re.call(m1, "GET", "/api/v1/projects/{pid}/jobs/"+re.job2, nil); code != 404 {
		t.Fatalf("someone else's job must look missing: %d", code)
	}
	// Lists hold only the member's own generations; operators see all.
	var list struct {
		Generations []struct {
			Job struct {
				ID string `json:"id"`
			} `json:"job"`
		} `json:"generations"`
	}
	_, body := re.call(m1, "GET", "/api/v1/projects/{pid}/generations", nil)
	_ = json.Unmarshal(body, &list)
	if len(list.Generations) != 1 || list.Generations[0].Job.ID != re.job1 {
		t.Fatalf("member list: %s", body)
	}
	list.Generations = nil
	_, body = re.call(op, "GET", "/api/v1/projects/{pid}/generations", nil)
	_ = json.Unmarshal(body, &list)
	if len(list.Generations) != 2 {
		t.Fatalf("operator list: %s", body)
	}
	// Downloads: owner and staff yes, other members no.
	if code, _ := re.call(m1, "GET", "/api/v1/projects/{pid}/artifacts/"+re.video1, nil); code != 200 {
		t.Fatalf("own video: %d", code)
	}
	if code, _ := re.call(m2, "GET", "/api/v1/projects/{pid}/artifacts/"+re.video1, nil); code != 404 {
		t.Fatalf("another member's video: %d", code)
	}
	if code, _ := re.call(re.tokens["viewer"], "GET", "/api/v1/projects/{pid}/artifacts/"+re.video1, nil); code != 200 {
		t.Fatalf("viewer download: %d", code)
	}
	// Cancel: not another member's job, but one's own.
	if code, _ := re.call(m1, "POST", "/api/v1/projects/{pid}/jobs/"+re.job2+"/cancel", nil); code != 404 {
		t.Fatalf("cancel of another member's job: %d", code)
	}
	if code, _ := re.call(m1, "POST", "/api/v1/projects/{pid}/jobs/"+re.job1+"/cancel", nil); code != 200 {
		t.Fatalf("cancel own job: %d", code)
	}
	var st string
	_ = re.rt.Deps.DB.QueryRow(context.Background(), `SELECT status FROM jobs WHERE id=$1`, re.job2).Scan(&st)
	if st != "ready" {
		t.Fatalf("job2 must be untouched, is %s", st)
	}
}

func TestPriorityIsClamped(t *testing.T) {
	re, c := newRoleEnv(t)
	defer c.stop(t)
	job := map[string]any{"job_type": "builtin.sleep", "params": map[string]any{"seconds": 1}, "priority": 9999}
	prio := func(tok string) int {
		code, body := re.call(tok, "POST", "/api/v1/projects/{pid}/jobs", job)
		if code != 201 {
			t.Fatalf("create: %d %s", code, body)
		}
		var out struct {
			IDs []string `json:"job_ids"`
		}
		_ = json.Unmarshal(body, &out)
		var p int
		_ = re.rt.Deps.DB.QueryRow(context.Background(), `SELECT priority FROM jobs WHERE id=$1`, out.IDs[0]).Scan(&p)
		return p
	}
	if p := prio(re.tokens["operator"]); p != 19 {
		t.Fatalf("operator priority = %d, want 19 (below the video band)", p)
	}
	if p := prio(re.tokens["admin"]); p != 100 {
		t.Fatalf("admin priority = %d, want 100", p)
	}
}

func TestInviteFlow(t *testing.T) {
	re, c := newRoleEnv(t)
	defer c.stop(t)
	admin := re.tokens["admin"]
	code, body := re.call(admin, "POST", "/api/v1/invites", map[string]any{"role": "member", "note": "for Sam"})
	if code != 201 {
		t.Fatalf("create invite: %d %s", code, body)
	}
	var inv struct {
		Code     string `json:"code"`
		JoinPath string `json:"join_path"`
	}
	_ = json.Unmarshal(body, &inv)
	if !strings.HasPrefix(inv.Code, "nvi_") || inv.JoinPath != "/join?code="+inv.Code {
		t.Fatalf("invite: %s", body)
	}
	// The code is never stored in the clear.
	var n int
	_ = re.rt.Deps.DB.QueryRow(context.Background(), `SELECT count(*) FROM user_invites WHERE note=$1 AND code_hash <> convert_to($2, 'UTF8')`, "for Sam", inv.Code).Scan(&n)
	if n != 1 {
		t.Fatal("invite row missing")
	}
	// Public check, no credentials needed.
	if code, _ := re.call("", "POST", "/api/v1/public/join/check", map[string]any{"code": inv.Code}); code != 200 {
		t.Fatalf("check: %d", code)
	}
	if code, _ := re.call("", "POST", "/api/v1/public/join/check", map[string]any{"code": "nvi_doesnotexist0000000000"}); code != 400 {
		t.Fatalf("bad code: %d", code)
	}
	join := func(user, pw string) int {
		c, _ := re.call("", "POST", "/api/v1/public/join", map[string]any{"code": inv.Code, "username": user, "display_name": "Sam", "password": pw})
		return c
	}
	if code := join("sam", "short"); code != 400 {
		t.Fatalf("weak password: %d", code)
	}
	if code := join("mem-one", "a long enough password"); code != 409 {
		t.Fatalf("taken username: %d", code)
	}
	if code := join("sam", "a long enough password"); code != 201 {
		t.Fatalf("join: %d (the failed attempts must not have used up the code)", code)
	}
	if code := join("sam2", "a long enough password"); code != 400 {
		t.Fatalf("a second use of the code: %d", code)
	}
	// Sam signs in, is a member, and can make their own API token.
	a := login(t, re.url, "sam", "a long enough password")
	var me struct {
		User struct{ Role string } `json:"user"`
	}
	if code := a.do("GET", "/api/v1/auth/me", nil, &me); code != 200 || me.User.Role != "member" {
		t.Fatalf("me: %d %+v", code, me)
	}
	var tok struct {
		Token string `json:"token"`
	}
	if code := a.do("POST", "/api/v1/tokens", map[string]any{"name": "Sam's laptop"}, &tok); code != 201 || !strings.HasPrefix(tok.Token, "nivra_pat_") {
		t.Fatalf("token: %d %+v", code, tok)
	}
	if code, _ := re.call(tok.Token, "GET", "/api/v1/projects/{pid}/limits", nil); code != 200 {
		t.Fatalf("limits with the new token: %d", code)
	}
	if code, _ := re.call(tok.Token, "GET", "/api/v1/projects/{pid}/workers", nil); code != 403 {
		t.Fatalf("a member must not list workers: %d", code)
	}
	// Revoking: disable the account and the token stops working.
	var users struct {
		Users []struct{ ID, Username string } `json:"users"`
	}
	_, ub := re.call(admin, "GET", "/api/v1/users", nil)
	_ = json.Unmarshal(ub, &users)
	var samID string
	for _, u := range users.Users {
		if u.Username == "sam" {
			samID = u.ID
		}
	}
	if code, _ := re.call(admin, "POST", "/api/v1/users/"+samID+"/status", map[string]any{"active": false}); code != 200 {
		t.Fatalf("disable: %d", code)
	}
	if code, _ := re.call(tok.Token, "GET", "/api/v1/projects", nil); code != 401 {
		t.Fatalf("disabled account's token still works: %d", code)
	}
	// An admin cannot lock themself out.
	_, ub = re.call(admin, "GET", "/api/v1/users", nil)
	var adminID string
	_ = json.Unmarshal(ub, &users)
	for _, u := range users.Users {
		if u.Username == "adm-one" {
			adminID = u.ID
		}
	}
	if code, _ := re.call(admin, "POST", "/api/v1/users/"+adminID+"/status", map[string]any{"active": false}); code != 409 {
		t.Fatalf("self-disable: %d", code)
	}
	// Expired and revoked invitations are refused.
	_, body = re.call(admin, "POST", "/api/v1/invites", map[string]any{"role": "viewer"})
	var inv2 struct {
		Code   string `json:"code"`
		Invite struct {
			ID string `json:"id"`
		} `json:"invite"`
	}
	_ = json.Unmarshal(body, &inv2)
	if code, _ := re.call(admin, "DELETE", "/api/v1/invites/"+inv2.Invite.ID, nil); code != 200 {
		t.Fatalf("revoke invite: %d", code)
	}
	if code, _ := re.call("", "POST", "/api/v1/public/join", map[string]any{"code": inv2.Code, "username": "late-one", "password": "a long enough password"}); code != 400 {
		t.Fatalf("revoked invite: %d", code)
	}
	_, body = re.call(admin, "POST", "/api/v1/invites", map[string]any{"role": "member"})
	var inv3 struct {
		Code string `json:"code"`
	}
	_ = json.Unmarshal(body, &inv3)
	if _, err := re.rt.Deps.DB.Exec(context.Background(), `UPDATE user_invites SET expires_at = now() - interval '1 minute'`); err != nil {
		t.Fatal(err)
	}
	if code, _ := re.call("", "POST", "/api/v1/public/join", map[string]any{"code": inv3.Code, "username": "late-two", "password": "a long enough password"}); code != 400 {
		t.Fatalf("expired invite: %d", code)
	}
}

// Two clients redeeming one code at the same time create exactly one account.
func TestInviteRedeemRace(t *testing.T) {
	re, c := newRoleEnv(t)
	defer c.stop(t)
	inv, _, err := re.rt.Auth.CreateInvite(context.Background(), "", auth.RoleMember, "race", 0)
	if err != nil {
		t.Fatal(err)
	}
	results := make(chan error, 6)
	for i := 0; i < 6; i++ {
		go func(i int) {
			_, err := re.rt.Auth.RedeemInvite(context.Background(), inv, fmt.Sprintf("racer-%d", i), "", "a long enough password")
			results <- err
		}(i)
	}
	ok := 0
	for i := 0; i < 6; i++ {
		if <-results == nil {
			ok++
		}
	}
	if ok != 1 {
		t.Fatalf("%d accounts were created from one single-use code", ok)
	}
}
