# Transcribe videos

1. Put your videos (`.mp4`, `.mkv`, `.mov`, `.webm`; subfolders are fine) in this folder.
2. Double-click `Transcribe.bat` (or run `python transcribe.py`).

Next to each video you get `<name>.txt` (what is visible: the training caption) and, when the video has
speech or embedded subtitles, `<name>.subtitles.txt`. Videos that already have a `.txt` are skipped, so you can
stop at any time and run it again to continue.

The first run installs everything by itself into a private environment (`%LOCALAPPDATA%\Nivra\transcribe-venv`):
PyTorch (GPU build with an NVIDIA card, otherwise CPU), the caption and speech libraries, and ffmpeg (winget).
Only Python 3.10–3.14 must already be installed.

Useful options: `--limit 100` (next 100 videos), `--overwrite`, `--subtitles off`, `--captioner qwen3vl` (better
captions, needs a GPU with ~8 GB). On a CPU expect about 2–3 minutes per video. Details:
[../docs/training/CAPTIONING.md](../docs/training/CAPTIONING.md).
