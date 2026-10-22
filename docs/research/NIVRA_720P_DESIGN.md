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
  difference loss + edge loss (stylized) + a hinge patch-GAN switched on late (section 9).
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

## 8. Stylized specialization (3D animation, anime, cartoon, 2D animation)

The shipped configs (`nivra_hd_tiny`, `nivra_medium`, `nivra_large`) target stylized content: `"domain": "stylized"`, `"styles"` (the canonical list
from `ai/nivra/data/styles.py`, `realistic` last and never the default), `"default_style": "anime"`. Configs without these keys load as
`domain: "general"`, `styles: []`, `default_style: null` and behave as before. This is a data and conditioning choice, not an architecture change:
`hd_config_hash` and the parameter counts are unchanged.

* **Style conditioning**: a style is a text prefix, `"<style> style. <prompt>"` (`apply_style`; skipped when the prompt already names a style alias or
  the style is `none`). Ingest stores a `style` per clip; the DiT trainer encodes `apply_style(caption, clip style)` and drops the tag for a
  `train.style_dropout` fraction of samples (0.1; a pure function of seed, epoch and clip id, so resume is exact) so the model also works untagged.
  Clips with `unspecified` style are trained on the plain caption.
* **HC-VAE edge loss**: `vae.edge_weight` (0.5 in the stylized configs, 0 = old loss) adds the L1 between horizontal/vertical finite differences of input
  and reconstruction (`edge_loss`) to push for crisp outlines instead of blurred line art. It is a training option (stripped from the architecture
  config), so VAE files and hashes are unaffected.
* **Data**: ingest `--profile stylized` relaxes the flat/static filters for flat colour areas and animation on 2s/3s and uses a scene threshold suited
  to hard cuts (see `docs/training/DATA_PIPELINE.md`). Only licensed data with `PROVENANCE.json` is ingested.
* **Bundles / inference**: `model_version.json` carries `domain`, `styles`, `default_style`; the loader exposes them on the pipeline. `ai.generate` takes
  `style` (canonical tag, alias, or `"none"`); omitted = the manifest `default_style`; unknown or unsupported = exit 2; small bundles reject it. The
  result summary and metadata contain `style` and `prompt_used` (the text actually encoded). Continuations reuse the stored, already-tagged prompt.
* **Evaluation**: `ai.evaluate` mode `adherence` on a stylized HD bundle runs the 24-prompt English set `ai/nivra/evaluation/stylized_prompts.json`
  (3 prompts for each of the 8 stylized styles) and reports plumbing statistics only (non-blank rate, motion, edge density). There is no detector for
  open-vocabulary stylized content, so no prompt-adherence or quality number is produced.
* **Not claimed**: nothing here has been trained on real stylized data; the tiny tests only show that tags, filters, losses and plumbing behave as
  specified. Output quality of trained models is unknown until training runs exist.

## 9. 2026 improvement pass (from `docs/research/IMPROVEMENT_RESEARCH_2026.md`)

All items are config-gated and backward compatible: a config without the new keys, and every bundle written before, loads and behaves as it did
(`hd_config_sha` of old configs is unchanged; the old tiny bundle regenerates bit-identical latents). Operational detail and the per-config table are in
`docs/training/TRAINING.md` (section "2026 improvement pass"). Versions: `nivra-hc-vae/2` (per-channel latent normalisation) and `nivra-dit/2`
(`frame_cond` / `fps_cond`) are only produced when those features are used; `/1` files stay readable.

* **VAE sharpness.** Hinge patch-GAN (small 3D discriminator, T=1 for images) from `vae_gan_start`, weight 0.05 x the VQGAN adaptive weight
  `||dL_rec/dW|| / ||dL_gan/dW||` at the decoder's last layer (computed on the loss-scaled graph, 0 if non-finite); discriminator trained from scratch,
  no licence issue. `edge_weight` defaults to 0.3 in stylized configs without an explicit value (the shipped ones set 0.5).
* **VAE schedule** (Wan recipe, no pretrained weights): image autoencoder at T=1 -> causal-3D low-resolution short clips with ~20 % image batches ->
  target-resolution fine-tune, GAN on in the last part. `train.vae_stages` carries it; `train-pipeline` scales it to `--vae-steps` (and scales
  `vae_gan_start` the same way).
* **Per-channel latent normalisation.** `[C]` shift / scale from encode statistics, stored in `hc_vae.json`, the safetensors metadata and the bundle
  config; `to_scaled` / `from_scaled` accept both forms, so the encode job, the sampler and the continuation re-encode need no change. The hd config
  hash ignores the calibration, so it does not change when statistics are filled in.
* **Temporal factor 4 or 8** (`vae.temporal_factor`; alias for `temporal_down`). **Decision for `nivra_base`: 4x, agreeing with the report.**
  Reasons: (1) Wan, HunyuanVideo, Open-Sora 2.0 and DC-AE-V all use 4x and only LTX uses 8x (with a denoising decoder Nivra does not have);
  (2) the VAE cannot be changed after latents are encoded, so the asymmetric risk favours the safer factor: 8x spans 4 drawings of anime on twos
  and loses fast cuts / 3D animation on ones; (3) at the base target the price is small: 256x448 x 17 frames goes from 14x8x3 = 336 to 14x8x5 = 560
  tokens (attention is cheap there; `shift_base_tokens` was moved 336 -> 560 so the shift curve is unchanged for the same clip). **Medium and Large keep
  8x**: at 720p x 8 s a 4x VAE doubles 28.5 k tokens to ~57 k, which full attention cannot afford on a 6 GB card (report item 15). No A/B reconstruction
  comparison has been run (no GPU, no real data): the report's advice to measure PSNR / edge F1 at 4x vs 8x on 3D-animation and fast-action clips before
  locking the VAE still stands, and either value is one config line (`temporal_factor`).
* **DiT curriculum.** Image-first stage (`frames: 1`; stills from the same causal VAE: a one-frame input encodes to exactly latent frame 0), then joint
  image + video stages (`image_fraction`, optional `buckets`). Per-resolution timestep shift was verified: `resolution_shift(tokens, base) =
  clamp(sqrt(tokens/base), 1, 6)` with tokens = latent T x H x W, used identically by the trainer (per micro-batch, so a T=1 still gets shift 1)
  and by inference (`HDPipeline`); Open-Sora 2.0 scales its shift with T x H x W the same way.
* **SRA** (self-representation alignment, no external model): block `student_layer` of the online DiT at noise t is pulled (cosine, through a small MLP
  projector) towards block `teacher_layer` of the EMA model at noise t x `t_ratio`. On by default for `nivra_base` (3 -> 8 of 12, weight 0.5, first
  60 k steps). Costs one extra no-grad forward of the EMA model per micro-batch. **REPA** (DINOv2 features, Apache-2.0 weights) would be a licensed
  opt-in teacher: the interface (`RepaHook`) and the config key exist, `enabled: true` is rejected, nothing is downloaded.
* **Diffusion forcing + conditioning frames.** A fraction (0.25 for base) of DiT micro-batches uses an independent logit-normal timestep per latent
  frame (the timestep embedding, adaLN modulation and gates become per-frame, same cost); for `frame_cond` models 15 % of batches also hold the first
  and/or last latent frame(s) clean with t = 0 and a learned flag embedding, excluded from the loss. Inference imposes continuation overlaps and refine
  anchors the same way (clean, masked) instead of re-noising them; older models fall back to the previous replacement scheme.
* **fp16 on Turing.** fp32 islands (embedders, adaLN, norms, gates, optional attention logits, loss) and a dynamic loss scaler that skips and logs
  overflow steps. Not yet run on a GPU.
* **Anime frame holds.** Ingest detects the drawing rate per clip (`hold`, `effective_fps`); `fps_cond` models condition on it; optional re-sampling at
  the drawing rate. OCR credit / subtitle filtering: not done (needs a new dependency).
* **EMA.** `train.ema_decay` (with a (1+n)/(10+n) warm-up); the pipeline exports the EMA weights as the bundle's DiT when it is on (it is on in every
  shipped config; `nivra_hd_tiny` uses 0.99, which only averages a handful of steps). Before this pass the pipeline had no EMA at all and exported the
  raw weights.
* **Not done from the report:** DC-AE-1.5 structured latent channels, denoising decoder, R1 penalty on the discriminator, register tokens, sandwich norm,
  post-hoc EMA, int8 deltas, few-step distillation, block-linear attention.
