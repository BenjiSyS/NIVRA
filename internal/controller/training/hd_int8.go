package training

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

// int8 deltas ("int8-rowscale/1", ai/nivra/distributed/hd/quant.py).
//
// For every base tensor NAME of shape S the update holds
//
//	NAME         I8   shape S           values q in [-127, 127]
//	NAME#scale   F32  shape [rows(S)]   per-row scale s >= 0, finite
//
// with rows(S) = 0 if numel(S) == 0, S[0] if rank >= 2, else 1. The dequantised value is float32(q)*s (a float32
// multiply, bit-identical to the Python side). The file size is exactly DataStart + sum(numel) + 4*sum(rows).
// Golden fixture shared with the Python tests: ai/tests/fixtures/hd_int8_golden.{safetensors,json}.

const (
	int8ScaleSuffix = "#scale"
	// Error feedback can add up to half a quantisation step of the previous round to an honest delta.
	int8Slack = 1.0 + 1.0/128
)

func int8Rows(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	switch {
	case n == 0:
		return 0
	case len(shape) >= 2:
		return shape[0]
	}
	return 1
}

func sameShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// checkDeltaStructureInt8 verifies the update has exactly the int8 layout above for the base's tensors.
func checkDeltaStructureInt8(upd, base *stHeader) string {
	if len(upd.Tensors) != 2*len(base.Tensors) {
		return fmt.Sprintf("update has %d tensors, an int8 update of this base has %d", len(upd.Tensors), 2*len(base.Tensors))
	}
	var want int64
	for name, bt := range base.Tensors {
		if strings.HasSuffix(name, int8ScaleSuffix) {
			return fmt.Sprintf("base tensor %q uses the reserved suffix %q", name, int8ScaleSuffix)
		}
		qt, ok := upd.Tensors[name]
		if !ok {
			return fmt.Sprintf("update lacks tensor %q", name)
		}
		if qt.DType != "I8" {
			return fmt.Sprintf("tensor %q is %s, int8 deltas must be I8", name, qt.DType)
		}
		if !sameShape(qt.Shape, bt.Shape) {
			return fmt.Sprintf("tensor %q has shape %v, the base has %v", name, qt.Shape, bt.Shape)
		}
		if qt.End-qt.Begin != qt.numel() {
			return fmt.Sprintf("tensor %q holds %d bytes for %d elements", name, qt.End-qt.Begin, qt.numel())
		}
		rows := int8Rows(bt.Shape)
		st, ok := upd.Tensors[name+int8ScaleSuffix]
		if !ok {
			return fmt.Sprintf("update lacks the scales of tensor %q", name)
		}
		if st.DType != "F32" {
			return fmt.Sprintf("scales of %q are %s, must be F32", name, st.DType)
		}
		if len(st.Shape) != 1 || st.Shape[0] != rows {
			return fmt.Sprintf("scales of %q have shape %v, want [%d]", name, st.Shape, rows)
		}
		if st.End-st.Begin != rows*4 {
			return fmt.Sprintf("scales of %q hold %d bytes for %d rows", name, st.End-st.Begin, rows)
		}
		want += qt.numel() + rows*4
	}
	if upd.Size != upd.DataStart+want {
		return fmt.Sprintf("update file is %d bytes, expected exactly %d", upd.Size, upd.DataStart+want)
	}
	return ""
}

// scanDeltaInt8 streams an int8 update (structure already checked): scale sanity, no -128, no non-zero value under a
// zero scale, finite dequantised values, the same per-tensor bounds as scanDelta (times int8Slack), not all zero.
func scanDeltaInt8(path string, h, base *stHeader, linfMax float64) (deltaStats, string, error) {
	var st deltaStats
	f, err := os.Open(path)
	if err != nil {
		return st, "", err
	}
	defer f.Close()
	names := make([]string, 0, len(base.Tensors))
	for n := range base.Tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	lim := linfMax * int8Slack
	buf := make([]byte, 1<<20)
	var total float64
	for _, name := range names {
		qt, sct := h.Tensors[name], h.Tensors[name+int8ScaleSuffix]
		n := qt.numel()
		if n == 0 {
			continue
		}
		rows := sct.numel()
		sraw := make([]byte, rows*4)
		if _, err := f.ReadAt(sraw, h.DataStart+sct.Begin); err != nil {
			return st, "", err
		}
		scales := make([]float32, rows)
		for i := range scales {
			s := math.Float32frombits(binary.LittleEndian.Uint32(sraw[4*i:]))
			if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
				return st, fmt.Sprintf("tensor %q: scale %d is NaN or Inf", name, i), nil
			}
			if s < 0 {
				return st, fmt.Sprintf("tensor %q: scale %d is negative", name, i), nil
			}
			scales[i] = s
		}
		cols := n / rows
		br := bufio.NewReaderSize(io.NewSectionReader(f, h.DataStart+qt.Begin, n), 1<<20)
		var sumsq, linf float64
		var row, inRow int64
		for left := n; left > 0; {
			c := int64(len(buf))
			if left < c {
				c = left
			}
			if _, err := io.ReadFull(br, buf[:c]); err != nil {
				return st, "", err
			}
			for _, b := range buf[:c] {
				q := int8(b)
				if q == -128 {
					return st, fmt.Sprintf("tensor %q contains the value -128 (int8 deltas use -127..127)", name), nil
				}
				if q != 0 {
					s := scales[row]
					if s == 0 {
						return st, fmt.Sprintf("tensor %q: non-zero values under a zero scale (row %d)", name, row), nil
					}
					v32 := float32(q) * s
					if math.IsInf(float64(v32), 0) {
						return st, fmt.Sprintf("tensor %q: dequantised value overflows float32", name), nil
					}
					v := float64(v32)
					st.Nonzero = true
					sumsq += v * v
					if a := math.Abs(v); a > linf {
						linf = a
					}
				}
				if inRow++; inRow == cols {
					inRow = 0
					row++
				}
			}
			left -= c
		}
		l2 := math.Sqrt(sumsq)
		if linf > lim {
			return st, fmt.Sprintf("tensor %q: max |delta| %.4g exceeds the bound %.4g", name, linf, lim), nil
		}
		if l2 > lim*math.Sqrt(float64(n)) {
			return st, fmt.Sprintf("tensor %q: delta L2 %.4g exceeds the bound %.4g", name, l2, lim*math.Sqrt(float64(n))), nil
		}
		total += sumsq
		if linf > st.Linf {
			st.Linf = linf
		}
	}
	st.L2 = math.Sqrt(total)
	if !st.Nonzero {
		return st, "update is all zeros", nil
	}
	return st, "", nil
}
