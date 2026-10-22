"""Transcribe every video in this folder.

Put videos (.mp4 .mkv .mov .webm, also in subfolders) next to this file and run:

    python transcribe.py            (or double-click Transcribe.bat on Windows)

For each video it writes, next to the video and with the same name:
  <name>.txt            what is visible (a caption for training)
  <name>.subtitles.txt  the spoken words or embedded subtitles (when there are any)

The first run installs everything it needs by itself into a private Python
environment (it does not touch your system Python): PyTorch (CUDA build when an
NVIDIA GPU is present, otherwise CPU), the caption/speech models' libraries and,
on Windows, ffmpeg via winget. Models are downloaded on first use.

Options are passed through to `nivra caption`, e.g.:
    python transcribe.py --limit 100       caption the next 100 videos, run again to continue
    python transcribe.py --overwrite       redo videos that already have a .txt
    python transcribe.py --subtitles off   only the visual caption
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
AI_DIR = REPO / "ai"
MARKER_VERSION = "2"  # bump to force a reinstall when the requirements change


def say(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def env_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Nivra"
    else:
        base = Path.home() / ".nivra"
    return base / "transcribe-venv"


def venv_python(d: Path) -> Path:
    return d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list, **kw) -> None:
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def torch_index() -> str:
    """CUDA wheels when an NVIDIA GPU is present (newer build for Turing and later), else CPU."""
    smi = shutil.which("nvidia-smi") or (r"C:\Windows\System32\nvidia-smi.exe" if os.name == "nt" else None)
    if smi and Path(smi).exists():
        try:
            out = subprocess.run([smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
            cap = float(out[0]) if out else 0.0
        except Exception:
            cap = 0.0
        return "https://download.pytorch.org/whl/" + ("cu128" if cap >= 7.5 else "cu126")
    return "https://download.pytorch.org/whl/cpu"


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    if os.name == "nt" and shutil.which("winget"):
        say("Installing ffmpeg (winget)")
        subprocess.run(["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--source", "winget"], check=False)
        links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
        if (links / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(links) + os.pathsep + os.environ.get("PATH", "")
            os.environ["NIVRA_FFMPEG"] = str(links / "ffmpeg.exe")
            os.environ["NIVRA_FFPROBE"] = str(links / "ffprobe.exe")
    if not shutil.which("ffmpeg") and not os.environ.get("NIVRA_FFMPEG"):
        sys.exit("!! ffmpeg is required. Install it (Windows: winget install Gyan.FFmpeg; Linux: apt install ffmpeg; "
                 "macOS: brew install ffmpeg), then run this again.")


def ensure_env() -> Path:
    """Create the private environment and install the requirements once."""
    if not (3, 10) <= sys.version_info[:2] <= (3, 14):
        sys.exit(f"!! Python 3.10-3.14 is required (this is {sys.version.split()[0]}). Install Python 3.12 and run again.")
    d = env_dir()
    py = venv_python(d)
    marker = d / ".nivra-transcribe-ready"
    want = f"{MARKER_VERSION} {torch_index()}"
    if py.exists() and marker.exists() and marker.read_text().strip() == want:
        return py
    if not py.exists():
        say(f"Creating a private Python environment in {d}")
        venv.EnvBuilder(with_pip=True).create(d)
    say("Installing packages (first run only; PyTorch is a large download)")
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "torch", "--index-url", torch_index()])
    run([py, "-m", "pip", "install", "-r", AI_DIR / "requirements.txt", "-r", AI_DIR / "requirements-caption.txt"])
    marker.write_text(want)
    return py


def main() -> int:
    if not (AI_DIR / "nivra" / "cli.py").exists():
        sys.exit(f"!! The Nivra 'ai' folder was not found next to this folder ({AI_DIR}). Keep transcribe/ inside the Nivra folder.")
    ensure_ffmpeg()
    py = ensure_env()
    videos = [p for p in HERE.rglob("*") if p.suffix.lower() in (".mp4", ".mkv", ".mov", ".webm") and p.is_file()]
    if not videos:
        print(f"No videos found in {HERE}. Put .mp4/.mkv/.mov/.webm files here (subfolders are fine) and run again.")
        return 0
    say(f"{len(videos)} video(s) in {HERE}")
    env = dict(os.environ, PYTHONPATH=str(AI_DIR), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    args = sys.argv[1:]
    # Reject anything that looks like a path, so the folder is always this one.
    if any(not a.startswith("-") and re.search(r"[\\/:]", a) for a in args):
        sys.exit("!! Options only; the videos are always taken from this folder.")
    try:
        return subprocess.run([str(py), "-m", "nivra.cli", "caption", str(HERE), *args], env=env).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
