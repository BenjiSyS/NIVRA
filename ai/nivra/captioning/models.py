"""Model wrappers (lazy heavy imports): Florence-2 for frame captions, faster-whisper for speech."""
from __future__ import annotations

import os
from pathlib import Path

from . import DEFAULT_MODEL, DEFAULT_MODEL_REVISION, INSTALL_CMD, WHISPER_MODEL, CaptionError, DependencyError


def models_dir() -> str | None:
    """NIVRA_MODELS_DIR if set, otherwise None (= the standard Hugging Face cache)."""
    v = os.environ.get("NIVRA_MODELS_DIR")
    return v or None


def _quiet_hf() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _missing(pkg: str) -> DependencyError:
    return DependencyError(f"{pkg} is not installed. Install the captioning dependencies with:\n  {INSTALL_CMD}\n"
                           "(or run: nivra caption --install)")


def pick_device(requested: str) -> str:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(requested)
    try:
        import torch
    except ImportError:
        raise DependencyError("PyTorch is not installed (see ai/requirements.txt / the Nivra installer)")
    cuda = torch.cuda.is_available()
    if requested == "cuda" and not cuda:
        raise CaptionError("--device cuda requested but CUDA is not available")
    return "cuda" if (requested == "cuda" or (requested == "auto" and cuda)) else "cpu"


def read_wav_f32(path: Path):
    """16 kHz mono 16-bit WAV -> float32 array in [-1, 1] (so faster-whisper never needs PyAV to decode)."""
    import wave

    import numpy as np
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise CaptionError("internal error: expected a mono 16-bit WAV")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


class Florence:
    """Florence-2 via the native transformers implementation (no trust_remote_code)."""

    def __init__(self, model_id: str | None = None, device: str = "auto"):
        _quiet_hf()
        try:
            import torch
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as e:
            if "Florence2" in str(e):
                raise DependencyError("the installed transformers is too old for native Florence-2. Upgrade with:\n  "
                                      + INSTALL_CMD) from e
            raise _missing("transformers") from e
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
        except Exception:
            pass
        self.torch = torch
        self.device = pick_device(device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        mid = model_id or DEFAULT_MODEL
        kw = {"cache_dir": models_dir()}
        if not Path(mid).is_dir() and mid == DEFAULT_MODEL and DEFAULT_MODEL_REVISION:
            kw["revision"] = DEFAULT_MODEL_REVISION
        try:
            self.processor = AutoProcessor.from_pretrained(mid, **kw)
            self.model = Florence2ForConditionalGeneration.from_pretrained(mid, dtype=self.dtype, **kw)
        except Exception as e:  # network, wrong repo layout, ...
            raise CaptionError(
                f"could not load Florence-2 model {mid!r} with the native transformers implementation: {e}\n"
                f"Use the default ({DEFAULT_MODEL}) or a local directory of a transformers-format checkpoint.") from e
        self.model.to(self.device).eval()

    def run(self, image_path: Path, task: str, max_new_tokens: int = 256) -> str:
        from PIL import Image
        torch = self.torch
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)
        if self.dtype != torch.float32:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3, do_sample=False)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=image.size)
        out = parsed.get(task, "") if isinstance(parsed, dict) else str(parsed)
        return str(out).strip()

    def caption_frames(self, frames: list[Path]) -> tuple[str, str | None, str | None, list[str]]:
        """Return (detailed middle caption, earliest short caption, latest short caption, all short captions)."""
        n = len(frames)
        mid = n // 2
        shorts: list[str | None] = [None] * n
        for i, f in enumerate(frames):
            if i != mid:
                shorts[i] = self.run(f, "<CAPTION>", 64)
        detailed = self.run(frames[mid], "<MORE_DETAILED_CAPTION>", 256)
        early = shorts[0] if n >= 3 else None
        late = shorts[-1] if n >= 3 else None
        return detailed, early, late, [s for s in shorts if s]


class Whisper:
    """faster-whisper 'small' speech-to-text (language auto, VAD on)."""

    def __init__(self, device: str = "auto", model_size: str = WHISPER_MODEL):
        _quiet_hf()
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise _missing("faster-whisper") from e
        try:
            dev = pick_device(device)
        except DependencyError:  # no torch: CTranslate2 can still run on CPU
            dev = "cpu"
        self._cls, self._size, self._root = WhisperModel, model_size, models_dir()
        self.device = dev
        self.model = self._load(dev)

    def _load(self, dev: str):
        return self._cls(self._size, device=dev, compute_type="float16" if dev == "cuda" else "int8",
                         download_root=self._root)

    def transcribe(self, wav: Path) -> list[str]:
        def go():
            segs, _info = self.model.transcribe(read_wav_f32(wav), vad_filter=True, language=None,
                                                condition_on_previous_text=False)
            lines = []
            for s in segs:
                if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                    continue
                t = s.text.strip()
                if t and (not lines or lines[-1] != t):
                    lines.append(t)
            return lines
        try:
            return go()
        except Exception:
            if self.device != "cuda":
                raise
            self.device = "cpu"  # CUDA libs for CTranslate2 missing: fall back
            self.model = self._load("cpu")
            return go()
