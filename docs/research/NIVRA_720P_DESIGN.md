# Nivra 720p design (Nivra Medium / Large)

Target from the master specification: **1280×720, 30 fps, 8 s = 240 model-generated frames**, prompt
adherence, temporal consistency, and roughly **30–40 minutes end-to-end on an RTX 2060 6 GB** (load +
text encoding + sampling + decoding). This document is the contract between the components; numbers
marked *estimate* are calculations, not measurements.

## 1. Pipeline

```
prompt ──► TextEncoder (pluggable) ──► text tokens [L, D_text] ─┐
                                                                ▼
noise ──► Nivra DiT (rectified flow, 3D RoPE, cross-attention) ──► latents ──► Nivra HC-VAE decoder ──► 241 frames @ 30 fps
```

## 2. Nivra HC-VAE ("decoder"), original, trained from scratch

* Causal 3D-convolutional VAE with **32× spatial and 8× temporal compression, 128 latent channels**
  (first frame encoded on its own so clips are 1 + 8k frames: 241 frames → 1 + 30 = 31 latent frames).
* 1280×720 is padded to 1280×736 → latent grid **40 × 23 × 31 = 28,520 tokens** (patch size 1).
* Decoder: residual 3D blocks + depth-to-space upsampling; optional timestep-conditioned final
  denoising stage. **Chunked causal temporal decoding** and **spatial tiling with overlap blending**
  so decoding fits in 6 GB.
* Losses: L1 + LPIPS-free perceptual proxy (original, no pretrained networks) + small KL + temporal
  difference loss; optional adversarial loss later.
* Versioned config + compatibility metadata (latent shape, scaling/shift factors, causal flag).

## 3. Text encoders (pluggable interface)

```python
class TextEncoder(Protocol):
    id: str            # e.g. "nivra-text-base/1" or "external:umt5-xl"
    dim: int           # embedding width D_text
    max_len: int
    def encode(self, prompts: list[str]) -> tuple[Tensor, Tensor]:  # emb [B,L,D] float32, mask [B,L] bool
```

* **Nivra Text (original)**: byte-level BPE tokenizer trained by Nivra (multilingual, EN/SV first-class,
  no unknown words) + transformer encoder (base ≈ 110 M params, d = 768, 12 layers). Training: (a)
  masked-language pretraining on a licensed text corpus, then (b) joint/contrastive alignment with video
  captions. Honest expectation: weak until trained on large corpora.
* **External encoders (third-party, opt-in)**: adapter that loads a locally stored, separately licensed
  encoder (e.g. a T5/UMT5 family encoder) in safetensors. Always recorded in the model manifest as
  `text_encoder: external:<name>@<sha256>` with its license; never relabelled as Nivra weights. Nivra never
  downloads one without the owner's approval.
* The generator projects any encoder's `D_text` to its own width, so checkpoints record which encoder
  they were trained with and refuse mismatches.

## 4. Nivra DiT generator

* Transformer over the 28.5 k latent tokens with **full spatio-temporal self-attention** (3D RoPE),
  cross-attention to text tokens, adaLN-Zero timestep conditioning, rectified-flow (velocity) objective,
  logit-normal timestep sampling with resolution-dependent shift.
* Variable resolution/duration via 3D RoPE and aspect buckets (256p/2 s … 720p/8 s).
* Configs:

| Variant | d | layers | heads | params (estimate) | fp16 weights | Purpose |
|---------|--:|-------:|------:|------------------:|-------------:|---------|
| Nivra Small (exists) | 192 | — | — | 2.4 M | — | 64×64 baseline |
| Nivra Medium | 1536 | 24 | 24 | ≈ 1.0 B | ≈ 2.0 GB | **720p/8 s on RTX 2060 (target 30–40 min)** |
| Nivra Large | 2304 | 32 | 32 | ≈ 3.5 B | ≈ 7 GB (≈ 14–15 GB packaged fp32) | ≥ 12 GB GPUs; 6 GB only with layer offload (slow) |

## 5. RTX 2060 inference budget (*estimate*, to be replaced by measurements)

Per denoising step for Medium at 28.5 k tokens: ≈ 2 · 1.0e9 · 28.5e3 (linear) + 4 · 24 · 1536 · 28.5e3²
(attention) ≈ 1.8e14 FLOPs; ×2 with CFG ≈ 3.6e14. At an effective 10–15 TFLOPS in fp16 on Turing
(memory-efficient SDPA kernel, no FlashAttention-2) that is ≈ 25–35 s/step, so 40 steps ≈ 17–24 min,
plus HC-VAE tiled decode (≈ 3–8 min) and text encoding/offload (≈ 1–3 min) → **≈ 25–35 min**.
Memory: weights 2 GB fp16 + activations for one 28.5 k-token block (attention via memory-efficient
kernel, MLP chunking) ≈ 2–3 GB; text encoder and VAE are offloaded to CPU when not in use.
Fewer steps (distilled/few-step sampling) is the main later speed-up.

## 6. Training reality

* Inference fits a 6 GB GPU; **training Medium/Large does not** (optimizer states alone exceed 6 GB).
  A 6 GB worker can train ≤ ~250 M-parameter models at low resolution with fp16, activation
  checkpointing and 8-bit optimizer states. Full Medium training needs ≥ 24 GB GPUs (or many GPUs with
  sharding) and millions of captioned clips.
* Plan: progressive training — VAE first (images + short clips), then DiT at 256p/2 s → 480p/4 s →
  720p/8 s, on licensed data from the ingestion pipeline. Distributed DiLoCo rounds apply to GPU workers
  that can hold the model.

## 7. Data pipeline (real video)

* **Folder ingestion**: `nivra data ingest <folder>` (via `python -m nivra.cli data-ingest`) takes videos
  plus captions (`clip.mp4` + `clip.txt` or a `captions.jsonl`) and a mandatory `PROVENANCE.json`
  (source, license, permission, collection date). Steps: ffprobe validation, scene splitting, aspect
  bucketing, resize/crop, quality filters (resolution, duration, motion, black/static detection),
  perceptual-hash deduplication, train/val split, sharded manifests. Optional latent precomputation with
  the HC-VAE into safetensors shards.
* **Public datasets**: a researched shortlist with licences is kept in `docs/research/DATASETS.md`;
  nothing is downloaded without the owner's explicit approval per dataset.
