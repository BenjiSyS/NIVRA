"""Nivra Remote Text-to-Video: run generation on a Nivra network.

The node sends the prompt to a Nivra controller, which queues it and hands
it to any connected worker computer with the AI executor (for example a GPU
PC). Many people and many ComfyUI installations can use the same network at
the same time; the controller's queue spreads the jobs over the workers.

Authentication uses a personal API token, read from the NIVRA_TOKEN
environment variable or from a token file. It is never a node input, so it is
never saved inside workflow files that people share.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch

TOKEN_FILES = [
    Path.home() / ".nivra" / "token",
]


def _token_file_candidates() -> list[Path]:
    paths = list(TOKEN_FILES)
    try:
        import folder_paths  # type: ignore

        paths.insert(0, Path(folder_paths.get_user_directory()) / "nivra_token.txt")
    except Exception:
        pass
    return paths


def load_token() -> str:
    tok = os.environ.get("NIVRA_TOKEN", "").strip()
    if tok:
        return tok
    for p in _token_file_candidates():
        try:
            tok = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if tok:
            return tok
    raise RuntimeError(
        "No Nivra API token found. Create one in Nivra Studio (Settings -> API tokens) or with "
        "`nivra admin token --username NAME`, then set the NIVRA_TOKEN environment variable or save it to "
        + " or ".join(str(p) for p in _token_file_candidates())
    )


class NivraClient:
    def __init__(self, base_url: str, token: str, timeout: float = 60):
        u = urllib.parse.urlparse(base_url.strip())
        local = u.hostname in ("localhost", "127.0.0.1", "::1") or (u.hostname or "").startswith(("192.168.", "10."))
        if u.scheme != "https" and not (u.scheme == "http" and local):
            raise ValueError("controller_url must use https:// (plain http is only allowed for local/LAN addresses)")
        self.base = base_url.strip().rstrip("/") + "/api/v1"
        self.token = token
        self.timeout = timeout

    def _req(self, method: str, path: str, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", "Bearer " + self.token)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read())["error"]["message"]
            except Exception:
                msg = str(e)
            raise RuntimeError(f"Nivra controller: HTTP {e.code}: {msg}") from None
        return raw

    def get(self, path):
        return json.loads(self._req("GET", path))

    def post(self, path, body=None):
        return json.loads(self._req("POST", path, body if body is not None else {}) or b"{}")

    def download(self, path, dst: Path):
        dst.write_bytes(self._req("GET", path))

    def project_id(self, name: str) -> str:
        projects = self.get("/projects")["projects"]
        for p in projects:
            if not name or p["name"] == name or p["id"] == name:
                return p["id"]
        raise RuntimeError(f"project {name!r} not found on the controller")


def _decode_mp4(path: Path) -> np.ndarray:
    try:
        from nivra.inference.video_io import decode_mp4  # type: ignore

        return decode_mp4(path)
    except ImportError:
        import av  # ComfyUI ships PyAV

        with av.open(str(path)) as c:
            return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def _interrupted() -> bool:
    try:
        import comfy.model_management as mm  # type: ignore

        return bool(mm.processing_interrupted())
    except Exception:
        return False


class NivraRemoteTextToVideo:
    """Generate on the Nivra network instead of this computer."""

    CATEGORY = "Nivra"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "mp4_path")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "controller_url": ("STRING", {"default": "https://studio.benjiserver.com"}),
            "prompt": ("STRING", {"multiline": True, "default": "a red circle moves from left to right"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "upscale": ("INT", {"default": 4, "min": 1, "max": 8}),
            "timeout_minutes": ("INT", {"default": 30, "min": 1, "max": 600}),
        }, "optional": {
            "project": ("STRING", {"default": ""}),
            "model_version": ("STRING", {"default": ""}),
        }}

    def run(self, controller_url, prompt, seed, steps, cfg_scale, upscale, timeout_minutes, project="", model_version=""):
        cl = NivraClient(controller_url, load_token())
        pid = cl.project_id(project)
        body = {"prompt": prompt, "seed": int(seed), "steps": int(steps), "cfg_scale": float(cfg_scale), "upscale": int(upscale)}
        if model_version.strip():
            body["model_version"] = model_version.strip()
        job = cl.post(f"/projects/{pid}/generate", body)["job_id"]
        print(f"[Nivra] queued remote generation {job}")
        deadline = time.time() + timeout_minutes * 60
        last = ""
        while True:
            if _interrupted():
                try:
                    cl.post(f"/projects/{pid}/jobs/{job}/cancel")
                finally:
                    raise RuntimeError("cancelled; the remote job was cancelled too")
            j = cl.get(f"/projects/{pid}/jobs/{job}")
            cur = j.get("current_attempt") or {}
            line = f"{j['status']} {cur.get('device_name', '')} {round(100 * (cur.get('progress') or 0))}%"
            if line != last:
                print(f"[Nivra] {line}")
                last = line
            if j["status"] == "completed":
                art = (j.get("result") or {}).get("artifact_id")
                if not art:
                    raise RuntimeError("the job finished without a video")
                break
            if j["status"] in ("failed", "cancelled"):
                raise RuntimeError(f"remote generation {j['status']}: {j.get('failure_reason', '')}")
            if time.time() > deadline:
                raise RuntimeError("timed out waiting for a Nivra worker (are AI workers connected and idle?)")
            time.sleep(2)
        out_dir = Path(tempfile.gettempdir()) / "nivra-remote"
        try:
            import folder_paths  # type: ignore

            out_dir = Path(folder_paths.get_output_directory())
        except Exception:
            pass
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"nivra_remote_{job[:8]}.mp4"
        cl.download(f"/projects/{pid}/artifacts/{art}", dst)
        frames = _decode_mp4(dst)
        return (torch.from_numpy(frames.astype(np.float32) / 255.0), str(dst))
