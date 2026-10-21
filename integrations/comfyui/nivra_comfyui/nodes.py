"""ComfyUI nodes for Nivra models.

Only capabilities that the Nivra package actually implements are exposed:
model loading, text-to-video, image-to-video (first-frame conditioning),
video continuation and MP4 export. Frames are returned as ComfyUI IMAGE
batches (float32 [T, H, W, 3] in 0..1), so they also work with any other
image/video node.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


def _import_nivra():
    try:
        import nivra  # noqa: F401
    except ImportError:
        extra = os.environ.get("NIVRA_AI_DIR")
        if not extra:
            # Development layout: <repo>/integrations/comfyui/nivra_comfyui -> <repo>/ai
            guess = Path(__file__).resolve().parents[3] / "ai"
            extra = str(guess) if (guess / "nivra").is_dir() else ""
        if extra and extra not in sys.path:
            sys.path.insert(0, extra)
    from nivra.inference import bundle as nb  # noqa: E402
    from nivra.inference import generate as ng  # noqa: E402
    from nivra.inference import video_io  # noqa: E402
    return nb, ng, video_io


try:  # Inside ComfyUI
    import folder_paths  # type: ignore

    _MODELS_DIR = os.path.join(folder_paths.models_dir, "nivra")
    os.makedirs(_MODELS_DIR, exist_ok=True)
    folder_paths.folder_names_and_paths.setdefault("nivra", ([_MODELS_DIR], set()))

    def _output_dir() -> str:
        return folder_paths.get_output_directory()
except ImportError:  # Unit tests / standalone use
    folder_paths = None
    _MODELS_DIR = os.environ.get("NIVRA_COMFY_MODELS", os.path.join(os.getcwd(), "models", "nivra"))

    def _output_dir() -> str:
        return os.environ.get("NIVRA_COMFY_OUTPUT", os.path.join(os.getcwd(), "output"))


def _bundles() -> list[str]:
    """Model bundles are directories containing weights.safetensors, model_version.json and vae.safetensors."""
    out = []
    if os.path.isdir(_MODELS_DIR):
        for name in sorted(os.listdir(_MODELS_DIR)):
            d = os.path.join(_MODELS_DIR, name)
            if all(os.path.isfile(os.path.join(d, f)) for f in ("weights.safetensors", "model_version.json", "vae.safetensors")):
                out.append(name)
    return out or ["(no Nivra bundles in models/nivra)"]


def _devices() -> list[str]:
    devs = ["auto", "cpu"]
    if torch.cuda.is_available():
        devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return devs


def _resolve_device(d: str) -> str:
    if d == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return d


def _frames_to_image(frames_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames_u8.astype(np.float32) / 255.0)


class NivraModel:
    """Loaded bundle plus the device it lives on."""

    def __init__(self, bundle, device: str, name: str):
        self.bundle, self.device, self.name = bundle, device, name


class NivraModelLoader:
    CATEGORY = "Nivra"
    RETURN_TYPES = ("NIVRA_MODEL",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bundle": (_bundles(),), "device": (_devices(),)}}

    def load(self, bundle, device):
        nb, _, _ = _import_nivra()
        path = os.path.join(_MODELS_DIR, bundle)
        if not os.path.isdir(path):
            raise ValueError(f"Nivra bundle not found: {path}. Put a bundle directory into models/nivra/.")
        dev = _resolve_device(device)
        try:
            b = nb.load_bundle(path, device=dev)
        except nb.BundleError as e:
            raise ValueError(f"Invalid or incompatible Nivra bundle '{bundle}': {e}") from e
        return (NivraModel(b, dev, bundle),)


class NivraTextToVideo:
    CATEGORY = "Nivra"
    RETURN_TYPES = ("IMAGE", "NIVRA_STATE")
    RETURN_NAMES = ("frames", "state")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("NIVRA_MODEL",),
            "prompt": ("STRING", {"multiline": True, "default": "a red circle moves from left to right"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "decode_tile": ("INT", {"default": 0, "min": 0, "max": 16}),
        }}

    def run(self, model, prompt, seed, steps, cfg_scale, decode_tile):
        _, ng, _ = _import_nivra()
        res = ng.generate(model.bundle, prompt, seed=seed, steps=steps, cfg_scale=cfg_scale, tile=decode_tile, device=model.device)
        if res.meta.get("unknown_words"):
            print(f"[Nivra] words outside this model's vocabulary were ignored: {res.meta['unknown_words']}")
        return (_frames_to_image(res.frames), res.state)


class NivraImageToVideo:
    CATEGORY = "Nivra"
    RETURN_TYPES = ("IMAGE", "NIVRA_STATE")
    RETURN_NAMES = ("frames", "state")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("NIVRA_MODEL",),
            "image": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "default": "a red circle moves from left to right"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
            "cfg_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
        }}

    def run(self, model, image, prompt, seed, steps, cfg_scale):
        _, ng, _ = _import_nivra()
        size = model.bundle.cfg["data"]["size"]
        img = image[0]  # first image of the batch, [H,W,3] 0..1
        if img.shape[0] != size or img.shape[1] != size:
            img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(size, size), mode="area")[0].permute(1, 2, 0)
        u8 = (img.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        res = ng.image_to_video(model.bundle, u8, prompt, seed=seed, steps=steps, cfg_scale=cfg_scale, device=model.device)
        return (_frames_to_image(res.frames), res.state)


class NivraVideoContinuation:
    CATEGORY = "Nivra"
    RETURN_TYPES = ("IMAGE", "NIVRA_STATE")
    RETURN_NAMES = ("frames", "state")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("NIVRA_MODEL",),
            "state": ("NIVRA_STATE",),
            "known_latent_frames": ("INT", {"default": 2, "min": 1, "max": 4}),
        }, "optional": {
            "prompt": ("STRING", {"multiline": True, "default": ""}),
        }}

    def run(self, model, state, known_latent_frames, prompt=""):
        _, ng, _ = _import_nivra()
        res = ng.continue_video(model.bundle, state, prompt=prompt or None, known_latent_frames=known_latent_frames, device=model.device)
        return (_frames_to_image(res.frames), res.state)


class NivraExportMP4:
    CATEGORY = "Nivra"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "fps": ("INT", {"default": 8, "min": 1, "max": 60}),
            "upscale": ("INT", {"default": 4, "min": 1, "max": 8}),
            "filename_prefix": ("STRING", {"default": "nivra"}),
        }}

    def save(self, frames, fps, upscale, filename_prefix):
        _, _, video_io = _import_nivra()
        out = Path(_output_dir())
        out.mkdir(parents=True, exist_ok=True)
        safe = "".join(c for c in filename_prefix if c.isalnum() or c in "-_") or "nivra"
        i = 1
        while (out / f"{safe}_{i:05d}.mp4").exists():
            i += 1
        path = out / f"{safe}_{i:05d}.mp4"
        u8 = (frames.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        video_io.write_mp4(u8, path, fps=fps, upscale=upscale)
        print(f"[Nivra] wrote {path} ({u8.shape[0]} frames)")
        return {"ui": {"text": [str(path)]}, "result": (str(path),)}


NODE_CLASS_MAPPINGS = {
    "NivraModelLoader": NivraModelLoader,
    "NivraTextToVideo": NivraTextToVideo,
    "NivraImageToVideo": NivraImageToVideo,
    "NivraVideoContinuation": NivraVideoContinuation,
    "NivraExportMP4": NivraExportMP4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NivraModelLoader": "Nivra Model Loader",
    "NivraTextToVideo": "Nivra Text-to-Video",
    "NivraImageToVideo": "Nivra Image-to-Video",
    "NivraVideoContinuation": "Nivra Video Continuation",
    "NivraExportMP4": "Nivra Export MP4",
}
