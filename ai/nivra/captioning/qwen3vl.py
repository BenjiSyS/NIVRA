"""Qwen3-VL (2B / 4B Instruct, Apache-2.0) as a video captioner: several frames of one clip go in as a multi-image
input (with their timestamps in the prompt) and a structured JSON description comes out (see structured.py)."""
from __future__ import annotations

import os
from pathlib import Path

from . import INSTALL_CMD, QWEN_MODEL, QWEN_MODEL_REVISION, CaptionError, DependencyError
from .models import _missing, _quiet_hf, models_dir, pick_device
from .structured import build_prompt


class Qwen3VL:
    def __init__(self, model_id: str | None = None, device: str = "auto"):
        _quiet_hf()
        try:
            import torch
            from transformers import AutoImageProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
        except ImportError as e:
            if "Qwen3VL" in str(e):
                raise DependencyError("the installed transformers is too old for Qwen3-VL. Upgrade with:\n  "
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
        # Turing has no bf16, so CUDA runs fp16; on the CPU bf16 halves the memory (2B: ~4.3 GB instead of ~8.6 GB)
        want = os.environ.get("NIVRA_QWEN_DTYPE", "")
        self.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            want, torch.float16 if self.device == "cuda" else torch.bfloat16)
        self.model_id = model_id or QWEN_MODEL
        kw = {"cache_dir": models_dir()}
        if not Path(self.model_id).is_dir() and self.model_id == QWEN_MODEL and QWEN_MODEL_REVISION:
            kw["revision"] = QWEN_MODEL_REVISION
        try:
            # Not AutoProcessor: Qwen3VLProcessor also builds a video processor, which needs torchvision. We feed frames
            # as images, so the image processor + tokenizer are all that is needed (see _inputs).
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, **kw)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kw)
            self.chat_template = self._read_chat_template()
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, dtype=self.dtype, **kw)
        except Exception as e:  # network, wrong repo layout, out of memory, ...
            raise CaptionError(f"could not load Qwen3-VL model {self.model_id!r}: {e}\n"
                               f"Use the default ({QWEN_MODEL}), the 4B variant or a local directory.") from e
        self.model.to(self.device).eval()

    def _read_chat_template(self) -> str | None:
        """chat_template.json of the checkpoint (the tokenizer config of this repo does not carry it)."""
        import json
        from huggingface_hub import hf_hub_download
        try:
            if Path(self.model_id).is_dir():
                f = Path(self.model_id) / "chat_template.json"
            else:
                rev = QWEN_MODEL_REVISION if self.model_id == QWEN_MODEL else None
                f = Path(hf_hub_download(self.model_id, "chat_template.json", revision=rev, cache_dir=models_dir()))
            return json.loads(f.read_text(encoding="utf-8"))["chat_template"]
        except Exception:
            return getattr(self.tokenizer, "chat_template", None)

    def _inputs(self, images, prompt: str):
        """input_ids / attention_mask / pixel_values / image_grid_thw for a one-turn chat with the images + prompt
        (what Qwen3VLProcessor does: one <|image_pad|> per merged patch)."""
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                  tokenize=False, chat_template=self.chat_template)
        img = self.image_processor(images=images, return_tensors="pt")
        grid = img["image_grid_thw"]
        merge = int(getattr(self.image_processor, "merge_size", 2)) ** 2
        pad = getattr(self.tokenizer, "image_token", None) or "<|image_pad|>"
        parts = text.split(pad)
        if len(parts) != len(images) + 1:
            raise CaptionError("internal error: the chat template did not produce one image placeholder per frame")
        text = parts[0] + "".join(pad * (int(g.prod()) // merge) + rest for g, rest in zip(grid, parts[1:]))
        tok = self.tokenizer(text, return_tensors="pt")
        return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"],
                "pixel_values": img["pixel_values"], "image_grid_thw": grid}

    def describe(self, frames: list[Path], times: list[float], max_new_tokens: int = 420, max_side: int = 448) -> str:
        """Raw model answer for the frames of one clip (frames are shrunk to ``max_side`` px to bound the token count)."""
        from PIL import Image
        torch = self.torch
        images = []
        for f in frames:
            with Image.open(f) as im:
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side))
                images.append(im.copy())
        inputs = {k: v.to(self.device) for k, v in self._inputs(images, build_prompt(len(images), times)).items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new = ids[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new, skip_special_tokens=True)[0].strip()
