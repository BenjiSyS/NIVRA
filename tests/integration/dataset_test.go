package integration

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"testing"

	"nivra/internal/controller/app"
	"nivra/internal/controller/auth"
	"nivra/internal/controller/core"
	"nivra/internal/shared/config"
	"nivra/internal/testutil"
)

// TestDatasetUploadAndRegister drives the operator upload API (chunked, with a
// resumed begin and a wrong-offset PUT) and the dataset registry over HTTP.
func TestDatasetUploadAndRegister(t *testing.T) {
	pool := testutil.DB(t)
	pool.Close()
	cfg := config.DefaultController()
	cfg.DataDir = t.TempDir()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.DatabaseURL = os.Getenv("NIVRA_TEST_DATABASE_URL")
	var tok, viewTok, pid string
	c := startController(t, cfg, func(rt *app.Runtime) {
		ctx := context.Background()
		for _, u := range []struct{ n, r string }{{"dsop", auth.RoleOperator}, {"dsview", auth.RoleViewer}} {
			if _, err := rt.Auth.CreateUser(ctx, u.n, "", "dataset test password", u.r); err != nil {
				t.Fatal(err)
			}
		}
		tok, _ = rt.Auth.TokenForUsername(ctx, "dsop", "t", 0)
		viewTok, _ = rt.Auth.TokenForUsername(ctx, "dsview", "t", 0)
		p, _, _ := rt.Projects.Create(ctx, core.System, "DS")
		pid = p.ID
	})
	defer c.stop(t)
	call := func(tk, method, path string, body []byte, out any) int {
		req, _ := http.NewRequest(method, c.url+"/api/v1/projects/"+pid+path, bytes.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+tk)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		b, _ := io.ReadAll(resp.Body)
		if out != nil {
			_ = json.Unmarshal(b, out)
		}
		return resp.StatusCode
	}
	type st struct {
		UploadID   string `json:"upload_id"`
		Offset     int64  `json:"offset"`
		State      string `json:"state"`
		ArtifactID string `json:"artifact_id"`
	}
	data := bytes.Repeat([]byte("nivra-shard-"), 1<<20) // 12 MB: two chunks
	sum := sha256.Sum256(data)
	sha := hex.EncodeToString(sum[:])
	begin, _ := json.Marshal(map[string]any{"kind": "dataset_shard", "sha256": sha, "size": len(data)})
	if code := call(viewTok, "POST", "/artifacts/uploads", begin, nil); code != 403 {
		t.Fatalf("viewer may not upload: %d", code)
	}
	var s1 st
	if code := call(tok, "POST", "/artifacts/uploads", begin, &s1); code != 200 || s1.State != "uploading" {
		t.Fatalf("begin: %d %+v", code, s1)
	}
	put := func(off int, b []byte, out *st) int {
		return call(tok, "PUT", "/artifacts/uploads/"+s1.UploadID+"?offset="+itoa(off), b, out)
	}
	var s2 st
	if code := put(0, data[:8<<20], &s2); code != 200 || s2.Offset != 8<<20 {
		t.Fatalf("chunk 1: %d %+v", code, s2)
	}
	if code := put(0, data[:8<<20], nil); code != 409 {
		t.Fatalf("replayed chunk must be an offset conflict, got %d", code)
	}
	// Beginning again resumes the same upload.
	var s3 st
	if code := call(tok, "POST", "/artifacts/uploads", begin, &s3); code != 200 || s3.UploadID != s1.UploadID || s3.Offset != 8<<20 {
		t.Fatalf("resume: %d %+v", code, s3)
	}
	if code := put(8<<20, data[8<<20:], &s2); code != 200 {
		t.Fatalf("chunk 2: %d", code)
	}
	var done st
	if code := call(tok, "POST", "/artifacts/uploads/"+s1.UploadID+"/complete", nil, &done); code != 200 || done.State != "complete" {
		t.Fatalf("complete: %d %+v", code, done)
	}
	reg, _ := json.Marshal(map[string]any{"name": "tiny", "kind": "raw",
		"shards": []map[string]any{{"idx": 0, "split": "train", "artifact_id": done.ArtifactID, "sha256": sha, "clips": 4, "bytes": len(data)}}})
	var ds struct {
		ID string `json:"id"`
	}
	if code := call(tok, "POST", "/datasets", reg, &ds); code != 201 || ds.ID == "" {
		t.Fatalf("register: %d", code)
	}
	if code := call(tok, "POST", "/datasets", reg, nil); code != 409 {
		t.Fatalf("duplicate register: %d", code)
	}
	var list struct {
		Datasets []struct {
			Dataset struct{ Name string } `json:"dataset"`
			Train   int                   `json:"train_shards"`
		} `json:"datasets"`
	}
	if code := call(viewTok, "GET", "/datasets", nil, &list); code != 200 || len(list.Datasets) != 1 || list.Datasets[0].Train != 1 {
		t.Fatalf("list: %d %+v", code, list)
	}
}

func itoa(n int) string { b, _ := json.Marshal(n); return string(b) }
