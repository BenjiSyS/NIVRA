"""MP4 writing / probing through ffmpeg + ffprobe subprocesses."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    pass


def write_mp4(frames: np.ndarray, path, fps: int = 8, upscale: int = 4) -> Path:
    """frames: uint8 [T,H,W,3]. H.264 / yuv420p. upscale = integer nearest-neighbour scale applied to the file only."""
    if shutil.which("ffmpeg") is None:
        raise FFmpegError("ffmpeg not found on PATH")
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    T, H, W, _ = frames.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-"]
    if upscale > 1:
        cmd += ["-vf", f"scale={W * upscale}:{H * upscale}:flags=neighbor"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "veryfast", "-movflags", "+faststart", str(path)]
    p = subprocess.run(cmd, input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise FFmpegError(p.stderr.decode(errors="replace")[:500])
    return path


def probe_mp4(path) -> dict:
    """Returns {'frames': counted frames, 'width', 'height', 'fps', 'codec'} using ffprobe -count_frames."""
    cmd = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
           "stream=nb_read_frames,width,height,r_frame_rate,codec_name,pix_fmt", "-of", "json", str(path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise FFmpegError(p.stderr.decode(errors="replace")[:500])
    s = json.loads(p.stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"frames": int(s["nb_read_frames"]), "width": s["width"], "height": s["height"],
            "fps": float(num) / float(den), "codec": s["codec_name"], "pix_fmt": s["pix_fmt"]}


def decode_mp4(path) -> np.ndarray:
    """Decode an MP4 back to uint8 [T,H,W,3] (used by evaluation of written files)."""
    info = probe_mp4(path)
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise FFmpegError(p.stderr.decode(errors="replace")[:500])
    arr = np.frombuffer(p.stdout, dtype=np.uint8)
    return arr.reshape(info["frames"], info["height"], info["width"], 3)


# ===================================================================== streaming (long videos)
class Mp4Writer:
    """Incremental H.264 writer: frames go through an ffmpeg stdin pipe as they are produced, so an hour of video
    is never held in memory. close() finalises the file (raises FFmpegError on failure); abort() deletes it."""

    def __init__(self, path, width: int, height: int, fps: float = 8, crf: int = 16, preset: str = "veryfast"):
        if shutil.which("ffmpeg") is None:
            raise FFmpegError("ffmpeg not found on PATH")
        if width % 2 or height % 2:
            raise FFmpegError(f"width/height must be even for yuv420p, got {width}x{height}")
        import tempfile
        self.path, self.width, self.height, self.frames = Path(path), int(width), int(height), 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._err = tempfile.TemporaryFile()
        fps_s = str(int(fps)) if float(fps) == int(fps) else repr(float(fps))
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
               "-r", fps_s, "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", preset,
               "-movflags", "+faststart", str(self.path)]
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err)
        self._closed = False

    def _fail(self):
        self._err.seek(0)
        msg = self._err.read().decode(errors="replace")[:500]
        raise FFmpegError(msg or "ffmpeg failed")

    def write(self, frames: np.ndarray) -> None:
        """frames: uint8 [T,H,W,3] (or a single [H,W,3])."""
        frames = np.ascontiguousarray(frames, dtype=np.uint8)
        if frames.ndim == 3:
            frames = frames[None]
        if frames.shape[1:] != (self.height, self.width, 3):
            raise FFmpegError(f"frame size {frames.shape[1:]} does not match writer {(self.height, self.width, 3)}")
        try:
            self._p.stdin.write(frames.tobytes())
        except (BrokenPipeError, OSError):
            self._fail()
        self.frames += frames.shape[0]

    def close(self) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        try:
            self._p.stdin.close()
        except OSError:
            pass
        rc = self._p.wait()
        try:
            if rc != 0:
                self._fail()
        finally:
            self._err.close()
        return self.path

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._p.stdin.close()
            except OSError:
                pass
            try:
                self._p.kill()
                self._p.wait()
            except OSError:
                pass
            self._err.close()
        try:
            self.path.unlink()
        except OSError:
            pass


class RawReader:
    """Streams decoded rgb24 frames of a video through ffmpeg. `start`/`count` select frames [start, start+count)
    (accurate time seek on a constant-frame-rate file), `size` rescales (bicubic)."""

    def __init__(self, path, width: int, height: int, fps: float, start: int = 0, count: int | None = None,
                 size: tuple | None = None):
        if shutil.which("ffmpeg") is None:
            raise FFmpegError("ffmpeg not found on PATH")
        import tempfile
        self.w, self.h = size if size else (width, height)
        cmd = ["ffmpeg", "-loglevel", "error"]
        if start > 0:
            cmd += ["-ss", f"{max(0.0, (start - 0.5) / fps):.6f}"]
        cmd += ["-i", str(path)]
        if count is not None:
            cmd += ["-frames:v", str(count)]
        vf = f"scale={self.w}:{self.h}:flags=bicubic" if size and size != (width, height) else None
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self._err = tempfile.TemporaryFile()
        self._p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._err, bufsize=0)
        self.frame_bytes = self.w * self.h * 3

    def read(self, out: np.ndarray | None = None):
        """Next frame as uint8 [H,W,3], or None at end of stream. With `out` (an [H,W,3] view) fills it in place."""
        buf = bytearray(self.frame_bytes) if out is None else None
        mv = memoryview(buf) if buf is not None else memoryview(out.reshape(-1))
        got = 0
        while got < self.frame_bytes:
            n = self._p.stdout.readinto(mv[got:])
            if not n:
                break
            got += n
        if got == 0:
            return None
        if got != self.frame_bytes:
            raise FFmpegError("truncated video stream")
        return out if out is not None else np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)

    def close(self) -> None:
        try:
            self._p.stdout.close()
        except OSError:
            pass
        try:
            self._p.kill()
        except OSError:
            pass
        self._p.wait()
        self._err.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def probe_fast(path) -> dict:
    """Like probe_mp4 but counts packets (no decoding): cheap on hour-long files. Raises FFmpegError."""
    cmd = ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
           "stream=nb_read_packets,width,height,r_frame_rate", "-of", "json", str(path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise FFmpegError(p.stderr.decode(errors="replace")[:500])
    try:
        s = json.loads(p.stdout)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return {"frames": int(s["nb_read_packets"]), "width": s["width"], "height": s["height"],
                "fps": float(num) / float(den)}
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
        raise FFmpegError(f"cannot probe {path}: {e}")
