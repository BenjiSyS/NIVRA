package training

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// mp4Info is what the structural MP4 check learned about a file.
type mp4Info struct {
	DurationSec float64
	Width       int
	Height      int
	Frames      int
	VideoTracks int
}

// maxMoovBytes bounds the metadata the controller will read into memory (a
// 30 minute 60 fps video has well under 10 MB of sample tables).
const maxMoovBytes = 64 << 20

type box struct {
	typ        string
	start, end int64 // whole box, including the header
	hdr        int64
}

func readBoxHeader(r io.ReaderAt, off, limit int64) (box, error) {
	var h [16]byte
	if limit-off < 8 {
		return box{}, errors.New("truncated box header")
	}
	if _, err := r.ReadAt(h[:8], off); err != nil {
		return box{}, err
	}
	size := int64(binary.BigEndian.Uint32(h[:4]))
	b := box{typ: string(h[4:8]), start: off, hdr: 8}
	switch size {
	case 0:
		size = limit - off // extends to the end of the enclosing box/file
	case 1:
		if limit-off < 16 {
			return box{}, errors.New("truncated 64-bit box header")
		}
		if _, err := r.ReadAt(h[8:16], off+8); err != nil {
			return box{}, err
		}
		size = int64(binary.BigEndian.Uint64(h[8:16]))
		b.hdr = 16
	}
	if size < b.hdr || off+size > limit {
		return box{}, fmt.Errorf("box %q has an invalid or truncated size", b.typ)
	}
	b.end = off + size
	return b, nil
}

// children lists the boxes directly inside [from, to).
func children(r io.ReaderAt, from, to int64) ([]box, error) {
	var out []box
	for off := from; off < to; {
		b, err := readBoxHeader(r, off, to)
		if err != nil {
			return out, err
		}
		out = append(out, b)
		off = b.end
	}
	return out, nil
}

func find(bs []box, typ string) (box, bool) {
	for _, b := range bs {
		if b.typ == typ {
			return b, true
		}
	}
	return box{}, false
}

func readAt(r io.ReaderAt, off int64, n int) ([]byte, error) {
	buf := make([]byte, n)
	if _, err := r.ReadAt(buf, off); err != nil {
		return nil, err
	}
	return buf, nil
}

// inspectMP4 checks that a file is a structurally complete MP4: a ftyp box,
// a non-empty mdat, a moov with a movie header of non-zero duration and at
// least one video track with samples whose total size fits inside mdat (a
// truncated upload fails here). It is a pure-Go stand-in for a quick ffprobe.
func inspectMP4(r io.ReaderAt, size int64) (mp4Info, error) {
	var info mp4Info
	top, err := children(r, 0, size)
	if err != nil {
		return info, fmt.Errorf("not a complete MP4 file: %w", err)
	}
	if len(top) == 0 || top[0].typ != "ftyp" {
		return info, errors.New("not an MP4 file (no ftyp box)")
	}
	moov, ok := find(top, "moov")
	if !ok {
		return info, errors.New("MP4 has no moov box (the file is incomplete)")
	}
	var mdatBytes int64
	for _, b := range top {
		if b.typ == "mdat" {
			mdatBytes += b.end - b.start - b.hdr
		}
	}
	if mdatBytes <= 0 {
		return info, errors.New("MP4 has no media data")
	}
	if moov.end-moov.start > maxMoovBytes {
		return info, errors.New("MP4 metadata is unreasonably large")
	}
	mchildren, err := children(r, moov.start+moov.hdr, moov.end)
	if err != nil {
		return info, fmt.Errorf("MP4 metadata is damaged: %w", err)
	}
	mvhd, ok := find(mchildren, "mvhd")
	if !ok {
		return info, errors.New("MP4 has no movie header")
	}
	body, err := readAt(r, mvhd.start+mvhd.hdr, int(min(mvhd.end-mvhd.start-mvhd.hdr, 120)))
	if err != nil || len(body) < 20 {
		return info, errors.New("MP4 movie header is damaged")
	}
	var timescale, duration uint64
	if body[0] == 1 {
		if len(body) < 32 {
			return info, errors.New("MP4 movie header is damaged")
		}
		timescale = uint64(binary.BigEndian.Uint32(body[20:24]))
		duration = binary.BigEndian.Uint64(body[24:32])
	} else {
		timescale = uint64(binary.BigEndian.Uint32(body[12:16]))
		duration = uint64(binary.BigEndian.Uint32(body[16:20]))
	}
	if timescale == 0 || duration == 0 {
		return info, errors.New("MP4 has zero duration")
	}
	info.DurationSec = float64(duration) / float64(timescale)
	var sampleBytes int64
	for _, t := range mchildren {
		if t.typ != "trak" {
			continue
		}
		w, h, frames, bytes, video, err := inspectTrack(r, t)
		if err != nil {
			return info, fmt.Errorf("MP4 track is damaged: %w", err)
		}
		if !video {
			continue
		}
		info.VideoTracks++
		info.Width, info.Height, info.Frames = w, h, frames
		sampleBytes += bytes
	}
	switch {
	case info.VideoTracks == 0:
		return info, errors.New("MP4 has no video track")
	case info.Frames <= 0:
		return info, errors.New("MP4 video track has no frames")
	case info.Width <= 0 || info.Height <= 0:
		return info, errors.New("MP4 video track has no picture size")
	case sampleBytes > mdatBytes:
		return info, errors.New("MP4 media data is shorter than its index says (truncated file)")
	}
	return info, nil
}

func inspectTrack(r io.ReaderAt, trak box) (w, h, frames int, sampleBytes int64, video bool, err error) {
	tc, err := children(r, trak.start+trak.hdr, trak.end)
	if err != nil {
		return
	}
	if tkhd, ok := find(tc, "tkhd"); ok {
		n := tkhd.end - tkhd.start - tkhd.hdr
		if n >= 8 {
			var tail []byte
			if tail, err = readAt(r, tkhd.end-8, 8); err != nil {
				return
			}
			w = int(math.Round(float64(binary.BigEndian.Uint32(tail[0:4])) / 65536))
			h = int(math.Round(float64(binary.BigEndian.Uint32(tail[4:8])) / 65536))
		}
	}
	mdia, ok := find(tc, "mdia")
	if !ok {
		return
	}
	mc, err := children(r, mdia.start+mdia.hdr, mdia.end)
	if err != nil {
		return
	}
	hdlr, ok := find(mc, "hdlr")
	if !ok {
		return
	}
	hb, err := readAt(r, hdlr.start+hdlr.hdr, int(min(hdlr.end-hdlr.start-hdlr.hdr, 12)))
	if err != nil || len(hb) < 12 || string(hb[8:12]) != "vide" {
		err = nil
		return
	}
	video = true
	minf, ok := find(mc, "minf")
	if !ok {
		return
	}
	minc, err := children(r, minf.start+minf.hdr, minf.end)
	if err != nil {
		return
	}
	stbl, ok := find(minc, "stbl")
	if !ok {
		return
	}
	sc, err := children(r, stbl.start+stbl.hdr, stbl.end)
	if err != nil {
		return
	}
	if stsz, ok := find(sc, "stsz"); ok {
		hdrBytes, e := readAt(r, stsz.start+stsz.hdr, int(min(stsz.end-stsz.start-stsz.hdr, 12)))
		if e != nil || len(hdrBytes) < 12 {
			err = errors.New("sample size table is damaged")
			return
		}
		fixed := int64(binary.BigEndian.Uint32(hdrBytes[4:8]))
		count := int64(binary.BigEndian.Uint32(hdrBytes[8:12]))
		frames = int(count)
		if fixed != 0 {
			sampleBytes = fixed * count
		} else {
			// Per-sample sizes: sum them, bounded by the box.
			avail := (stsz.end - stsz.start - stsz.hdr - 12) / 4
			if count > avail {
				err = errors.New("sample size table is truncated")
				return
			}
			var tbl []byte
			if tbl, err = readAt(r, stsz.start+stsz.hdr+12, int(count*4)); err != nil {
				return
			}
			for i := int64(0); i < count; i++ {
				sampleBytes += int64(binary.BigEndian.Uint32(tbl[i*4 : i*4+4]))
			}
		}
	} else if stz2, ok := find(sc, "stz2"); ok {
		hb, e := readAt(r, stz2.start+stz2.hdr, int(min(stz2.end-stz2.start-stz2.hdr, 12)))
		if e == nil && len(hb) >= 12 {
			frames = int(binary.BigEndian.Uint32(hb[8:12]))
		}
	}
	return
}
