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

INSTALL_CMD = "python -m pip install -r ai/requirements-caption.txt"


class CaptionError(RuntimeError):
    """A per-video or setup failure with a message that is safe to show to the user."""


class DependencyError(CaptionError):
    pass
