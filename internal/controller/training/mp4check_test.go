package training

import (
	"bytes"
	"encoding/binary"
	"os"
	"testing"
)

func mkBox(typ string, parts ...[]byte) []byte {
	body := bytes.Join(parts, nil)
	out := make([]byte, 8, 8+len(body))
	binary.BigEndian.PutUint32(out, uint32(8+len(body)))
	copy(out[4:], typ)
	return append(out, body...)
}

func be32(v uint32) []byte { b := make([]byte, 4); binary.BigEndian.PutUint32(b, v); return b }

// synthMP4 builds a minimal structurally valid MP4 with n frames of size each.
func synthMP4(n, frameBytes int, timescale, duration uint32, w, h int, handler string) []byte {
	ftyp := mkBox("ftyp", []byte("isom"), be32(512), []byte("isomiso2"))
	mdat := mkBox("mdat", make([]byte, n*frameBytes))
	mvhd := mkBox("mvhd", be32(0), be32(0), be32(0), be32(timescale), be32(duration), make([]byte, 80))
	tkhd := mkBox("tkhd", be32(0), make([]byte, 72), be32(uint32(w)<<16), be32(uint32(h)<<16))
	hdlr := mkBox("hdlr", be32(0), be32(0), []byte(handler), make([]byte, 12), []byte{0})
	stsz := mkBox("stsz", be32(0), be32(uint32(frameBytes)), be32(uint32(n)))
	stbl := mkBox("stbl", stsz)
	minf := mkBox("minf", stbl)
	mdia := mkBox("mdia", hdlr, minf)
	trak := mkBox("trak", tkhd, mdia)
	moov := mkBox("moov", mvhd, trak)
	return bytes.Join([][]byte{ftyp, mdat, moov}, nil)
}

func TestInspectMP4(t *testing.T) {
	good := synthMP4(48, 1000, 1000, 2000, 1280, 720, "vide")
	info, err := inspectMP4(bytes.NewReader(good), int64(len(good)))
	if err != nil {
		t.Fatalf("good file rejected: %v", err)
	}
	if info.Frames != 48 || info.Width != 1280 || info.Height != 720 || info.DurationSec != 2 {
		t.Fatalf("info %+v", info)
	}
	cases := map[string][]byte{
		"truncated":      good[:len(good)-30],
		"no moov":        good[:bytes.Index(good, []byte("moov"))-4],
		"zero duration":  synthMP4(48, 1000, 1000, 0, 1280, 720, "vide"),
		"no frames":      synthMP4(0, 1000, 1000, 2000, 1280, 720, "vide"),
		"audio only":     synthMP4(48, 1000, 1000, 2000, 1280, 720, "soun"),
		"zero size":      synthMP4(48, 1000, 1000, 2000, 0, 0, "vide"),
		"not mp4":        bytes.Repeat([]byte{0x42}, 400),
		"empty":          nil,
		"index too big":  bytes.Replace(good, be32(1000), be32(5000), 1), // first occurrence is inside stsz? checked below
		"garbage header": append([]byte{0, 0, 0, 4, 'f', 't', 'y', 'p'}, good[8:]...),
	}
	for name, data := range cases {
		if name == "index too big" {
			// Make the declared sample size larger than the media data.
			data = synthMP4(48, 1000, 1000, 2000, 1280, 720, "vide")
			i := bytes.Index(data, []byte("stsz"))
			binary.BigEndian.PutUint32(data[i+8:], 5000)
		}
		if _, err := inspectMP4(bytes.NewReader(data), int64(len(data))); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func TestVideoParams(t *testing.T) {
	long := make([]rune, 1000)
	for i := range long {
		long[i] = 'ö'
	}
	ok := []string{`{}`, `{"prompt":"a cat","name":"seg_001"}`, `{"prompt":"` + string(long) + `","name":"a.b-c_d"}`}
	for _, s := range ok {
		if err := videoParams([]byte(s)); err != nil {
			t.Errorf("%.40s: %v", s, err)
		}
	}
	bad := []string{`{"prompt":""}`, `{"prompt":"` + string(long) + `x"}`, `{"name":"../../evil"}`, `{"name":"a/b"}`, `{"name":"a\\b"}`, `{"name":".hidden"}`, `{"name":""}`, `{"name":5}`, `[1]`}
	for _, s := range bad {
		if err := videoParams([]byte(s)); err == nil {
			t.Errorf("%.40s: expected an error", s)
		}
	}
}

// TestInspectRealMP4 checks the parser against an ffmpeg-made file when
// NIVRA_TEST_MP4 points at one (CI without ffmpeg skips it).
func TestInspectRealMP4(t *testing.T) {
	p := os.Getenv("NIVRA_TEST_MP4")
	if p == "" {
		t.Skip("NIVRA_TEST_MP4 not set")
	}
	f, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	st, _ := f.Stat()
	info, err := inspectMP4(f, st.Size())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%+v", info)
	if info.Frames < 1 || info.Width != 320 || info.Height != 240 || info.DurationSec < 2.5 {
		t.Fatalf("unexpected %+v", info)
	}
	// Chop the tail: must be rejected.
	if _, err := inspectMP4(f, st.Size()-2000); err == nil {
		t.Fatal("truncated real file accepted")
	}
}
