"""Video captioning: a description (.txt) and spoken/embedded subtitle text (.subtitles.txt) per video.

Heavy dependencies (transformers, faster-whisper) are imported lazily; see requirements-caption.txt.
"""
from __future__ import annotations

STYLES = ["anime", "3d animation", "cartoon", "2d animation", "cel-shaded 3d", "pixel art",
          "claymation", "painterly", "realistic"]
STYLE_CHOICES = ["auto", *STYLES, "none"]
SUBTITLE_MODES = ["auto", "embedded", "speech", "off"]
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm")

# Native transformers implementation of Microsoft's Florence-2-large (MIT), converted by the
# Hugging Face team. The original microsoft/Florence-2-large repo needs trust_remote_code.
DEFAULT_MODEL = "florence-community/Florence-2-large"
# Pinned commit of DEFAULT_MODEL (verified on Python 3.14, transformers 5.20.2; see docs/training/CAPTIONING.md).
DEFAULT_MODEL_REVISION: str | None = "4271c66b88cdbc05735372ec13b2360108de5317"
WHISPER_MODEL = "small"

# Qwen3-VL-2B-Instruct (Apache-2.0, official Qwen org, verified on the model card: license apache-2.0; code repo
# QwenLM/Qwen3-VL is Apache-2.0). 4B is the fallback for more detail. Pinned commits as verified on 2026-10-22.
QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_MODEL_REVISION: str | None = "89644892e4d85e24eaac8bacfd4f463576704203"
QWEN_MODEL_4B = "Qwen/Qwen3-VL-4B-Instruct"
CAPTIONERS = ["auto", "florence", "qwen3vl"]

INSTALL_CMD = "python -m pip install -r ai/requirements-caption.txt"


class CaptionError(RuntimeError):
    """A per-video or setup failure with a message that is safe to show to the user."""


class DependencyError(CaptionError):
    pass
