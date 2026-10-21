package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// The upload client continues from the server's offset, survives one failed
// chunk, and completes with the full content.
func TestUploadFileResumes(t *testing.T) {
	payload := make([]byte, 20<<20+123)
	for i := range payload {
		payload[i] = byte(i * 7)
	}
	var stored []byte
	preExisting := 9 << 20 // a previous run already delivered 9 MB
	stored = append(stored, payload[:preExisting]...)
	failedOnce, completed := false, false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		st := func() map[string]any {
			return map[string]any{"upload_id": "u1", "offset": len(stored), "size": len(payload), "chunk_size": 8 << 20, "state": "uploading"}
		}
		switch {
		case r.Method == "POST" && r.URL.Path == "/api/v1/projects/p/artifacts/uploads":
			json.NewEncoder(w).Encode(st())
		case r.Method == "PUT":
			off, _ := strconv.Atoi(r.URL.Query().Get("offset"))
			if !failedOnce && off > preExisting {
				failedOnce = true
				http.Error(w, "boom", 500)
				return
			}
			if off != len(stored) {
				w.WriteHeader(409)
				json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"message": "offset"}, "status": st()})
				return
			}
			b, _ := io.ReadAll(r.Body)
			stored = append(stored, b...)
			json.NewEncoder(w).Encode(st())
		case r.Method == "GET":
			json.NewEncoder(w).Encode(st())
		case r.Method == "POST":
			completed = true
			json.NewEncoder(w).Encode(map[string]any{"upload_id": "u1", "state": "complete"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	path := filepath.Join(t.TempDir(), "s.bin")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	r := &remote{base: srv.URL + "/api/v1", hc: srv.Client(), token: "t"}
	id, err := r.uploadFile(context.Background(), "p", "dataset_shard", path, "ignored", int64(len(payload)))
	if err != nil || id != "u1" || !completed {
		t.Fatalf("id=%q completed=%v err=%v", id, completed, err)
	}
	if string(stored) != string(payload) {
		t.Fatalf("stored %d bytes, content differs from the file", len(stored))
	}
}

func TestReadDatasetManifest(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "dataset.json"), []byte(`{"fingerprint":"abc","width":448,"height":256,"frames":17,"fps":8,
		"shards":[{"file":"shard-val-00000.safetensors","split":"val","clips":2,"sha256":"x"},{"file":"shard-train-00000.safetensors","split":"train","clips":9,"sha256":"y"}]}`), 0o600)
	m, err := readDatasetManifest(dir)
	if err != nil || m.Shards[0].Split != "train" || m.Encoder["stage"] == nil {
		t.Fatalf("%+v %v", m, err)
	}
	os.WriteFile(filepath.Join(dir, "dataset.json"), []byte(`{"shards":[{"file":"../x","split":"train","clips":1}]}`), 0o600)
	if _, err := readDatasetManifest(dir); err == nil {
		t.Fatal("path traversal in shard file accepted")
	}
}
