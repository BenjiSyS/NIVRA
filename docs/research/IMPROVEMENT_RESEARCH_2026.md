# Nivra: what to change before training starts (research, October 2026)

Scope: Nivra is an original, from-scratch text-to-video system for stylized content (anime, 3D animation, cartoon). Volunteers train it on consumer GPUs; the owner's main card is an RTX 2060 6 GB (Turing: fp16 only, no bf16, no FP8, no FlashAttention-2). This report compares the current design with published 2024-2026 practice and ranks changes by impact against effort.

Method: web search over papers, technical reports and READMEs, checked against the Nivra source (`ai/nivra/compression/hc_vae.py`, `ai/nivra/models/dit.py`, `ai/nivra/training/hd_trainer.py`, `ai/nivra/distributed/hd/*`, `ai/configs/*.json`). Licence statements come from model cards or secondary sources as noted. **Verify every licence against the official LICENSE file before shipping anything.** Where evidence is thin or comes only from the authors themselves, the text says so.

What Nivra already does right (keep it): rectified flow with logit-normal t and a resolution-dependent shift (`resolution_shift`); QK-norm (RMSNorm); 3D RoPE; adaLN-Zero; EMA plus a held-out flow loss; outer Nesterov SGD at lr 0.7 / momentum 0.9 (`distributed/hd/aggregate.py`), the DiLoCo default; robust (trimmed-mean, clipped) aggregation; caption dropout (0.1); style-tag dropout; a causal VAE with streaming state; VAE tiling; int8 weight-only inference.

---

## 0. Reality check on compute (it shapes every recommendation)

| Reference | Model | Compute | Source |
|---|---|---|---|
| Open-Sora 2.0 | 11B, initialised from FLUX | ~4,160 H200-days (~$200k) | https://arxiv.org/abs/2503.09642 |
| SANA-Video | ~2B linear DiT | 64 H100 x 12 days (~768 H100-days) | https://arxiv.org/abs/2509.24695 |
| Reducio-DiT | image-conditioned, 1K video | 3.2K A100-hours | https://arxiv.org/abs/2411.13552 |
| PixArt-alpha (image) | 0.6B | 675 A100-days | https://arxiv.org/abs/2310.00426 |
| "Stretching Each Dollar" (image) | 1.16B sparse DiT | ~$1.9k, 2.6 days on 8xH100 (paper) | https://arxiv.org/abs/2407.15811 |

As a rough order of magnitude, an RTX 2060 delivers about 1-3% of an H100's useful fp16 training throughput once its 6 GB of memory forces checkpointing and offload. **One H100-day is roughly 1-3 months of a single 2060.** Even the cheapest from-scratch video results (hundreds of H100-days) therefore mean tens to hundreds of 2060-years. Three consequences follow:

1. The 213M **base** model at 256p is the only target the owner can reach alone. Medium (932M) and Large (2.8B) need a real volunteer pool, and should be presented that way.
2. Every technique that cuts the steps needed to converge (REPA or SRA, image-first training, a better-structured latent, dense captions) is worth more to Nivra than any architecture tweak.
3. Whatever must be original should be kept small and data-efficient. Text encoding and captioning are one-time preprocessing costs, so they can use larger licensed models without touching the training budget.

---

## Executive summary: top 15 changes, ranked by impact/effort for Nivra

| # | Change | Impact | Effort | Why (key evidence) |
|---|---|---|---|---|
| 1 | **Add an adversarial (patch-GAN) loss to the HC-VAE, switched on late (after L1/perceptual converge), and turn on `edge_weight` for stylized data** | Very high | M | No 32x video VAE in the literature reaches usable sharpness from L1+KL alone; LTX-Video, Wan, HunyuanVideo and Cosmos all use GAN fine-tuning. Thin anime line art is the first thing blur destroys. `edge_weight` defaults to 0.0 today. |
| 2 | **Train the VAE image-first, then inflate to video at low resolution with few frames, then fine-tune at target size (the Wan-VAE recipe), using a joint image+video mix** | High | S-M | The Wan 2.1 report (2503.20314) trains a 2D image VAE first, inflates it to causal 3D, then fine-tunes. It converges far faster than training on video from scratch. |
| 3 | **Reconsider the 8x temporal factor: use 4x temporal (32x32x4, as in Open-Sora 2.0 Video DC-AE and DC-AE-V), or keep 8x only with fps-normalised anime sampling** | High | M | Of the high-compression VAEs, only LTX uses 8x temporal, and it needs a denoising decoder. DC-VideoGen reports that earlier 32x VAEs had poor reconstruction and poor long-video generalisation, and chose 4x temporal. Anime held frames help but do not remove the risk. |
| 4 | **Make the latent space easier to model: per-channel normalisation (not one scalar shift/scale), plus DC-AE-1.5-style structured channels or VA-VAE-style alignment** | High | M | DC-AE 1.5 (2508.00413) and "Reconstruction vs Generation" (2501.01423): more latent channels improve reconstruction but slow diffusion convergence, worst for small DiTs. A 128-channel latent with a 213M DiT is in exactly that danger zone. |
| 5 | **Add a representation-alignment loss to DiT training: SRA (no external model, keeps weights fully original) by default, REPA with DINOv2 (Apache-2.0) as an opt-in add-on** | Very high | S-M | REPA: >17x faster convergence on ImageNet (2410.06940). SRA (ICLR 2026, 2505.02831) is roughly comparable without any external encoder. This is the cheapest large convergence win available. |
| 6 | **Image-first DiT pretraining on stylized stills (256p), then joint image+video at low resolution, then raise resolution and duration** | Very high | S | PixArt-alpha decomposition; Open-Sora 2.0 found 256p training ~40x cheaper than 768p and learned motion there. Anime stills are plentiful and cheap. |
| 7 | **Replace Florence-2 frame captions with a video-native VLM: Qwen3-VL-2B/4B-Instruct (Apache-2.0), structured dense captions plus a short caption, with mixed lengths during training** | High | M | PixArt-alpha and Open-Sora 2.0 (LLaVA-Video, then Qwen) both credit dense recaptioning. Merged frame captions cannot describe motion, which is the core of video. Avoid Qwen2.5-VL-3B (Qwen research licence, non-commercial). |
| 8 | **Ship a licensed text encoder add-on as the recommended default for quality (umT5-base/XL, Apache-2.0; or Qwen3-0.6B/1.7B hidden states, Apache-2.0), cache embeddings offline; keep NivraText as the "fully original" track** | High | S-M | Imagen: text encoder scale mattered more than denoiser scale. A 110M MLM trained only on captions sees too little language to generalise to novel prompts. Cached embeddings make encoder size irrelevant to the 6 GB budget. |
| 9 | **Hold-aware anime data handling: detect "on 2s/3s" holds, sample training clips at the drawing rate (or condition on effective fps), dedupe near-identical frames** | Medium-High | S | About half the frames in 24 fps anime are holds (2111.12792 removed ~49% as duplicates). Training on padded holds wastes temporal capacity and teaches stutter. |
| 10 | **Train a fraction of DiT steps with per-frame independent noise levels (diffusion forcing)** | Medium-High | S | Enables autoregressive extension and better seam conditioning for long video (SkyReels-V2 2504.13074, Diffusion Forcing 2407.01392) at nearly zero extra cost. It strengthens the existing draft-to-refine pipeline instead of replacing it. |
| 11 | **Quantise pseudo-gradients to int8 now and 4-bit later with error feedback, and stream fragments (Streaming DiLoCo); weight by tokens processed** | Medium | M | Streaming DiLoCo (2501.18512) reports 4-bit outer gradients with no loss; INTELLECT-1 used int8. This halves or quarters Pi and residential uplink load compared with fp16 deltas. |
| 12 | **fp16-on-Turing hardening: keep adaLN modulation, timestep MLP, RMSNorm statistics, softmax and the loss in fp32; dynamic loss scaling; register/sink tokens optional** | Medium | S | SANA 1.5 hit fp16 overflow in attention logits and needed QK-norm (already present). Turing has no bf16, so overflow handling must be explicit. |
| 13 | **Few-step distillation after the base model exists (rCM, DMD2 or shortcut-model fine-tune) plus training-free step caching (TeaCache-style)** | High at inference | M-L | rCM (2510.08431): 1-4 steps, 15-50x faster than the teacher on Wan. TeaCache: ~1.6-2x at small quality cost. Turing gets no FP8, and SageAttention only has slower Triton paths there. |
| 14 | **Evaluation: held-out flow loss at fixed t-bins as the primary signal; a small stylized prompt suite with VLM-judged adherence; DINOv2 frame-consistency and optical-flow motion scores; JEDi only if licences allow** | Medium | S | SD3 (2403.03206) found validation loss tracks human preference. FVD needs huge sample counts; JEDi needs ~16% of the samples (2410.05203). |
| 15 | **Linear or windowed attention for the 720p stage (SANA-Video style block-linear, or local-window plus a few global layers) instead of full 3D attention** | Medium (720p only) | L | At 720p and multi-second length, full attention dominates cost. SANA-Video reports 16x lower latency than Wan-1.3B with similar quality. Defer until 256-480p works. |

Do-not-do list (low value or high risk now): MMDiT dual-stream (doubles parameters, little proven win below 1B); a U-shaped DiT rewrite; DisTrO/DeMo (changes the optimizer family, more complexity than the bandwidth problem justifies); training a VAE at 720p before the 256p loop works; autoregressive self-forcing training before a strong bidirectional base exists.

---

## 1. VAE

### What the best open video VAEs do

| VAE | Spatial x temporal | Latent ch | Pixels per DiT token | Notes / source |
|---|---|---|---|---|
| Wan 2.1 VAE | 8x8x4 | 16 | 16x16x4 (2x2 patch) | Causal 3D, RMSNorm instead of GroupNorm for causal caching, 127M params. Image VAE first, inflated to 3D, then video fine-tune (with GAN). https://arxiv.org/abs/2503.20314 |
| Wan 2.2 VAE (TI2V-5B) | 16x16x4 | 48 (model card) | 32x32x4 (2x2 patch) | Runs 720p on a 4090. https://github.com/Wan-Video/Wan2.2 |
| HunyuanVideo VAE | 8x8x4 | 16 | 16x16x4 | Causal 3D, LPIPS + GAN, image+video mix. https://arxiv.org/abs/2412.03603 |
| HunyuanVideo 1.5 | 16x16x4 (secondary source) | 32 | — | 8.3B DiT, sparse SSTA attention. https://arxiv.org/abs/2511.18870 |
| LTX-Video VAE | 32x32x8 | 128 | 32x32x8 (patch 1) | 1:192 compression; **the decoder also performs the final denoising step** to recover detail. https://arxiv.org/abs/2501.00103 |
| Open-Sora 2.0 Video DC-AE | 32x32x4 | 128 | 32x32x4 (patch 1) | Initialised from the HunyuanVideo VAE; 5.2x training and >10x inference speedup over the 8x VAE. https://arxiv.org/abs/2503.09642 |
| DC-AE-V (DC-VideoGen) | 32x / 64x spatial, 4x temporal, chunk-causal | — | — | Reports that earlier 32x video AEs had low reconstruction quality and poor long-video generalisation; chunk-causal design fixes it. https://arxiv.org/abs/2509.25182 |
| Cosmos tokenizer | 8x8x8 / 8x16x16 | 16 (continuous) | — | NVIDIA Open Model License (commercial use allowed, custom terms). https://huggingface.co/nvidia/Cosmos-Tokenizer-CI8x8 |
| Reducio-VAE | extreme, image-conditioned | — | — | Encodes motion only, conditioned on a content image; 64x fewer latents than a 2D VAE. https://arxiv.org/abs/2411.13552 |

**Assessment of Nivra's 32x/8x x128ch with patch 1.** This is exactly LTX-Video's geometry (32x32x8 pixels per token, 128 dims, 1:192). It is a legitimate, published design point, and for Nivra it has real advantages: few tokens, so attention is cheap and 6 GB GPUs stay usable. But the evidence also says:

- **Every team that used 32x spatial added something to recover detail**: LTX moved final denoising into the decoder, Open-Sora 2.0 started from a pretrained 8x VAE, and DC-AE-V redesigned the temporal structure. Nivra trains from scratch with no GAN and no pretrained perceptual network. That combination has no published precedent for reaching good sharpness. **The VAE is the single largest risk in the project.**
- **8x temporal is the outlier.** Open-Sora 2.0, DC-AE-V, Wan and Hunyuan all use 4x. Anime "on twos" at 24 fps means an 8x latent frame spans 4 drawings, which is workable, but fast cuts, smears and 3D-animated content (often on ones) will suffer.
- **High channel counts slow diffusion convergence.** DC-AE 1.5 (https://arxiv.org/abs/2508.00413) and VA-VAE/LightningDiT (https://arxiv.org/abs/2501.01423) both show reconstruction improving with channels while generation gets harder, most of all for small DiTs. DC-AE 1.5's fix: train the AE so early channels carry structure and later ones carry detail (random channel-subset reconstruction), then put extra diffusion weight on the structure channels. Their f64c128 trained 4x faster than f32c32 for equal or better quality.

**Recommendation (ranked):**
1. Keep 32x spatial and 128 channels (it suits 6 GB inference). **Make the temporal factor configurable and train a 4x-temporal variant for comparison** at 256p (`temporal_down: [2, 3]`). If held-out reconstruction (PSNR, edge F1) on 3D-animation and fast-action clips is clearly worse at 8x, ship 4x. Cost: doubles latent T, so about 2x attention tokens on the time axis.
2. **Add DC-AE-1.5 structured-latent training:** during VAE training, randomly zero the trailing channel groups and reconstruct from the leading ones. Cheap, from-scratch compatible, no external weights.
3. **Per-channel latent normalisation.** `latent_shift`/`latent_scale` are currently scalars. Wan normalises each of its 16 channels with stored mean/std. With 128 channels of very different variance, a scalar scale leaves some channels badly conditioned for the shifted timestep schedule. Store `[C]` vectors in the shard manifest.
4. **Optional, Nivra-specific: a denoising decoder (LTX trick).** Train the decoder to accept slightly noisy latents (add noise at small t during VAE fine-tuning, condition on t). It makes the decoder tolerant of DiT errors, which matter more for a small DiT. Effort M.

### Losses that matter, and licensing

- **GAN loss** (patch discriminator, hinge or non-saturating), **switched on only after reconstruction stabilises** (Hunyuan, Wan and LTX all add it in later stages; starting it early destabilises training). 3D or (2+1)D discriminator for temporal realism. Typical weight 0.05-0.1 with an adaptive weight, as in VQGAN. **This is the most important missing piece.** A discriminator trained from scratch raises no licensing issue.
- **Perceptual loss.** LPIPS uses ImageNet-trained VGG/AlexNet weights. The LPIPS code is BSD-2, but the backbone weights are trained on ImageNet, whose terms are non-commercial research. Options that avoid this:
  - Nivra's own perceptual proxy (current), combined with a GAN, is defensible.
  - **DINOv2 features as a perceptual or alignment loss.** DINOv2 code and weights are Apache-2.0 (relicensed from CC-BY-NC in 2023; verify at https://github.com/facebookresearch/dinov2). Used only as a training-time loss, its weights are not shipped. Label it as an "external training-time teacher" for the "original weights" policy.
  - Avoid DINOv3 (custom licence) and anything CLIP/OpenAI-derived unless the licence is checked.
- **Frequency/edge losses for line art:** an FFT-magnitude loss or Sobel edge loss. Nivra has `edge_loss` but `edge_weight=0.0` by default. Turn it on (0.1-0.5) for stylized data.
- **KL:** keep it tiny (1e-6, as now). Several 2025 works use almost no KL plus latent noise augmentation or alignment instead.
- **Temporal loss:** the current frame-difference term is fine. A temporal discriminator does more.

### Training order and data
- Wan recipe: (a) 2D image AE at 256p; (b) inflate to causal 3D, train on low-res short clips (e.g. 128-256p, 17-33 frames); (c) fine-tune on target resolution and longer clips with the GAN on. Joint batches of images (T=1) and videos throughout. Hunyuan used roughly 1:4 image:video.
- Since Nivra stores DiT latents as the posterior mean, keep the encoder **frozen** once DiT latent shards are produced. Every VAE change invalidates all shards. **Lock the VAE before large-scale volunteer encoding.** This is irreversible in practice.

### Pretrained VAE as a labelled add-on
If the owner allows a licensed add-on, the fastest route to quality is to start from or adopt the **Wan 2.1 VAE (Apache-2.0)** or **Wan 2.2 VAE (Apache-2.0)**. Open-Sora 2.0 itself bootstrapped its DC-AE from Hunyuan's VAE. This conflicts with the "original weights" goal, so offer it only as an optional "compat" track.

---

## 2. DiT architecture at 200M-1B

| Choice | Evidence | Verdict for Nivra |
|---|---|---|
| Cross-attention vs MMDiT vs single-stream | PixArt and Wan use cross-attention; SD3/FLUX use MMDiT; Z-Image (6B, https://arxiv.org/abs/2511.22699) and Open-Sora 2.0's single-stream blocks concatenate text and visual tokens. No clean small-scale ablation shows MMDiT beating cross-attention below 1B. | **Keep cross-attention.** It is cheaper (text tokens do not enter self-attention) and the text embeddings can be cached. |
| QK-norm | SANA 1.5 (https://arxiv.org/abs/2501.18427): fp16 attention logits overflowed without it. | Already present. Keep. |
| Sandwich norm (pre+post) | Lumina-Next (https://arxiv.org/abs/2406.18583) uses it for stability at scale. | Optional; cheap insurance for fp16. Low priority. |
| Register tokens | "Vision Transformers Need Registers" (2309.16588); 2026 papers report registers help DiT convergence (https://arxiv.org/abs/2605.16147, https://arxiv.org/abs/2605.05206). Evidence is recent and mostly pixel-space. | Low-cost experiment: 4-8 learned tokens. Not a priority. |
| RoPE | 3D RoPE is standard (Wan, Hunyuan). Already present. For resolution and duration generalisation, scale the RoPE positions to a reference grid when changing stage (interpolation). | Keep; add position scaling across stages. |
| Windowed or sparse attention | Training-free sparse kernels (Sparse VideoGen 2502.01776, STA, Radial) give 1.5-2x on big models at 720p. SANA-Video's block-linear attention is trained in from the start. | At 256p (~1-3K tokens) full attention is fine. Revisit for 720p (#15). |
| Hourglass / U-shaped DiT | HDiT (2401.11605) shines in pixel space. With a 32x latent the token count is already small. | **Not worth it** with 32x latents. |
| Token masking / deferred masking | "Stretching Each Dollar": deferred masking (patch-mixer, then mask 75%) beat downscaling the model under equal FLOPs. | Worth an experiment for the image-pretraining phase (S-M effort). Less proven for video. |
| Width vs depth | LightningDiT/RAE: high-dimensional latents need a wider model or a wide head (DDT head). A 213M DiT at dim ~768 on a 128-channel latent is input-bottlenecked. | Make the patch-embed and final layer generous; consider dim ≥ latent_channels x 6. |

---

## 3. Training recipe for small compute

1. **Image-first, then video, in progressive stages** (PixArt-alpha 2310.00426; Open-Sora 2.0; Wan). Stage A: images at 256p (T=1 through the same causal VAE). Stage B: joint image+video at 256p, short clips. Stage C: longer clips. Stage D: higher resolution. Open-Sora 2.0 spent about 54% of its budget at 256p, and 768p cost ~40x more per sample.
2. **Buckets:** aspect-ratio and duration buckets, with images as one-frame clips. Already partly present through `bucket`.
3. **Timestep shift by token count:** already present (`resolution_shift`). Open-Sora 2.0 scales alpha with T x H x W, as Nivra does. Verify the base-token constant at each stage, and evaluate with a matched shift at inference.
4. **Representation alignment (the biggest cheap win):**
   - REPA (https://arxiv.org/abs/2410.06940): align an early-to-middle DiT block's tokens (through a small MLP) to DINOv2 features of the clean frames. >17x faster on ImageNet. Precompute the DINOv2 features once per clip into the shards. Apache-2.0 weights, but an external teacher, so it must be labelled.
   - **SRA** (https://arxiv.org/abs/2505.02831, ICLR 2026): aligns an earlier layer at higher noise to a later layer at lower noise of an EMA copy, with no external model. Reported comparable to REPA. **Fits the originality constraint.** Cost: one extra forward of the EMA model on part of the batch (~20-30% step time), which can be limited to the first part of training.
   - Video variants (VideoREPA 2505.23656) target physics fine-tuning, not pretraining speed. Low priority.
   - REPA-E (2504.10483) trains the VAE end-to-end with REPA. Too complex for now.
5. **EMA:** keep it, but with only a few thousand outer steps a plain 0.9999 decay may lag. Use a warm-up EMA (decay ramps with step) or post-hoc EMA (Karras EDM2), which lets you choose the EMA length after training.
6. **Loss weighting:** logit-normal (0, 1) plus shift is the SD3 optimum (https://arxiv.org/abs/2403.03206). Keep it.
7. **CFG dropout:** 10% (as now). Use a learned null embedding or zeros consistently. Drop the style tag separately (already done).
8. **Data quality filtering (Open-Sora 2.0 pipeline):** drop clips under 2 s, low bitrate, fps under 16, extreme aspect ratios; split at scenes; filter by motion score (optical flow), blur (Laplacian variance), OCR text area, and camera jitter. Aesthetic scorers are usually CLIP-based, so check licences (LAION aesthetic MLP is MIT but sits on OpenAI CLIP weights, also MIT). **Filtering out subtitles and credits matters for anime:** OCR text area is cheap with PaddleOCR (Apache-2.0).
9. **Synthetic data:** "Stretching Each Dollar" mixed synthetic images into a 37M-image set. For Nivra, rendered 3D-animation clips (Blender, CC0 assets) are a legally clean source of perfectly labelled motion.
10. **fp16 on Turing:** dynamic GradScaler; fp32 master weights (8-bit AdamW states are fine); compute adaLN modulation, the timestep embedding, norms and softmax in fp32; clip gradients at 1.0. If loss spikes appear, add sandwich norm.

---

## 4. Anime and stylized specialisation

- **Datasets and licensing.** There is no clean, commercially usable anime video dataset. Sakuga-42M (2405.07425) was **withdrawn** from arXiv and was research-only. AniSora (Bilibili, https://arxiv.org/abs/2412.10255) built 10M clips from 1M videos, but released only weights (Apache-2.0 for V2) and a benchmark, not the data. Tarsier2-Recap-585K says Apache-2.0 on the card but restricts use to academic research. **The owner's own downloads carry copyright risk in most jurisdictions.** Text-and-data-mining exceptions vary (EU DSM Art. 4 allows TDM with an opt-out; the US relies on fair-use case law, unsettled for generative training). Recommendation: keep a per-source provenance manifest (already partly present via `dataset_id`), honour robots/TDM opt-outs, and prefer CC-licensed animation (Blender open movies such as *Sintel*, *Spring* and *Sprite Fright* are CC-BY), the owner's own renders, and synthetic 3D renders.
- **Frame holds ("on 2s/3s").** About half of 24 fps anime frames are duplicates (2111.12792 removed ~49%). Recommendation: (a) detect holds per clip with a small pixel-difference threshold in Lab space; (b) estimate the "drawing rate" and either resample to it or pass `effective_fps` as a conditioning signal (micro-conditioning, as SDXL does for size); (c) never interpolate holds away. Timing is style. At inference, a "hold on 2s" post-process exists in the community (ComfyUI AIMZ node) and is trivial.
- **Line-art preservation:** GAN + edge loss in the VAE (section 1), and a high-frequency-aware evaluation (edge F1 between input and reconstruction). Flat colour regions compress very well at 32x, which partly justifies the high compression for 2D anime; lines are the bottleneck.
- **Style conditioning:** already has style tags plus dropout. Add a **source/medium tag** (2D cel, 3D CG, cartoon, chibi) and **quality/era tags**, as NovelAI and Animagine-style image models do. Keep tags in a fixed prefix so they can be dropped.
- **Tags vs prose:** community anime image models (for example the Anima model card) train on a mix of Danbooru tags, prose, and both, and recommend at least two sentences for prose prompts. No controlled study exists, so treat this as practitioner evidence. Recommendation: train on a mix (about 40% prose dense caption, 30% short prose, 20% tags plus short prose, 10% tags only) so either prompt style works. Tags can come from an open tagger, but check the licence (WD tagger models are Apache-2.0 on the card; verify).

---

## 5. Captioning (6-8 GB GPU or CPU)

| Model | Video-native | Fits 6 GB? | Licence | Notes |
|---|---|---|---|---|
| Florence-2 (current) | No (frames) | Yes | MIT | Good per-frame detail, no motion. |
| **Qwen3-VL-2B / 4B-Instruct** | Yes (timestamps, temporal reasoning) | 2B fp16 ~5 GB; 4B in 4-bit ~3-4 GB | **Apache-2.0** (model cards, per aggregators; verify) | Best fit. Released October 2025. |
| Qwen2.5-VL-3B | Yes | Yes | **Qwen Research licence (non-commercial)** | **Avoid**; the 7B is Apache-2.0 but too large for 6 GB at fp16. |
| Qwen2.5-VL-7B | Yes | 4-bit ~5.5 GB (tight) | Apache-2.0 | Fallback. |
| InternVL3-1B/2B | Yes | Yes | MIT code; LLM part inherits Qwen2.5 terms | Check the LLM component licence. |
| LLaVA-Video-7B | Yes | 4-bit only | Apache-2.0 code; Qwen2 base | Used by Open-Sora 2.0 for 256p data. |
| Tarsier2-7B | Yes, strong descriptions | 4-bit only | Unclear for weights | Do not depend on it. |
| VideoChat-Flash, Apollo | Yes | 2B variants exist | Mixed / research | Lower priority. |

Recommendations:
- Use **Qwen3-VL-4B-Instruct (4-bit) or 2B (fp16)** with 8-16 sampled frames per clip, a structured prompt (subject, appearance, action and motion with direction and speed, camera motion, setting, art style or medium, shot type) and JSON output. Keep Whisper for dialogue context, but **do not put transcribed speech into the visual caption**: it teaches the model to render text.
- Generate **two captions per clip** (short ~20 words, dense ~80-150 words) and sample between them during training. This handles short user prompts.
- Speed: on a 2060, expect a few seconds per clip for 2B and roughly 10-20 s for 4B 4-bit with 8 frames. For 100k clips, that is days to weeks. Distribute captioning as a volunteer job type (the encode-job infrastructure already exists).
- Optionally rewrite user prompts at inference into the training caption style with a small LLM (Qwen3-1.7B, Apache-2.0). Wan and Open-Sora both use prompt extension.

---

## 6. Text encoder

- Evidence: Imagen (https://arxiv.org/abs/2205.11487) found scaling the text encoder more impactful than scaling the denoiser. "Scaling Down Text Encoders" (CVPR 2025, https://arxiv.org/abs/2503.19897) distilled T5-XXL into T5-base (50x smaller) with comparable image quality, so **a ~200M encoder can work, but only when it is distilled from or pretrained on broad language**, not trained only on a caption corpus.
- A **110M MLM trained only on Nivra's captions** will understand the caption distribution but generalise poorly to unseen words, compositions, counting and negation. Expect weak prompt adherence for real users. Two original-weights ways to improve it: (a) pretrain NivraText on a large openly licensed text corpus (for example FineWeb-Edu, ODC-By, or Wikipedia, CC-BY-SA) before caption MLM; (b) train it contrastively against video latents (CLIP-style, own weights) in addition to MLM. Both are M effort and still trail a licensed encoder.
- Licensed add-on candidates (embed offline, so size costs only preprocessing time):

| Encoder | Licence | Used by | Notes |
|---|---|---|---|
| **umT5-base / XL** | Apache-2.0 | Wan 2.1 (umT5-XXL) | Multilingual (Nivra has Swedish prompts). XL encoder ~1.2B: fp16 on CPU or GPU for caching. |
| **Qwen3-0.6B / 1.7B (base)** hidden states | Apache-2.0 | Z-Image uses Qwen3-4B | Decoder LLM features with a learned refiner; strong composition. |
| T5Gemma, Gemma-2-2B | Gemma Terms of Use (use restrictions, flow-down) | SANA, Lumina 2 | Avoid for a "clean" licence story. |
| CLIP / OpenCLIP | MIT / various | SD1/SDXL | Weak at composition; not needed. |

Recommendation: keep NivraText as the "fully original" path, and **make umT5-base (Apache-2.0, multilingual, 768-wide, which matches `text_dim: 768`) the recommended licensed add-on**, because it is a drop-in replacement for width. Train the DiT from day one against cached embeddings of **one** encoder. Switching later costs a re-adaptation phase.

---

## 7. Inference speed on 6 GB GPUs

| Technique | Turing support | Realistic gain | Source |
|---|---|---|---|
| Few-step distillation (rCM, DMD2, shortcut models) | Yes (it is just fewer steps) | 10-25x fewer NFEs (50 to 2-4 steps); quality close to the teacher | rCM https://arxiv.org/abs/2510.08431, DMD2 https://arxiv.org/abs/2405.14867, Shortcut https://arxiv.org/abs/2410.12557 |
| CFG distillation (guidance-embedded) | Yes | 2x (one forward instead of two) | FLUX-dev style |
| TeaCache / FasterCache | Yes (training-free) | 1.5-2.2x at small quality loss | https://arxiv.org/abs/2411.19108 |
| SageAttention | **Partial**: sm75 CUDA kernels removed upstream, Triton path only, unofficial builds | ≤1.3x on Turing; not worth the dependency | https://github.com/thu-ml/SageAttention |
| FP8 | **No** (needs Ada/Hopper) | — | — |
| int8 weight-only | Yes (already present) | Memory only; slight slowdown | — |
| VAE tiling / streaming decode | Yes (already present) | Memory only | — |
| Sparse attention (STA, Sparse VideoGen) | Kernels target Hopper/Ampere | Small at 256p | 2502.01776 |

Order: (1) make 20-30-step Euler/AB2 sampling reliable on the base model; (2) add TeaCache-style skipping (S effort); (3) after the base converges, distil to 4-8 steps. Shortcut-model training can be folded into pretraining from the start (it conditions on step size), which avoids a separate distillation phase. Worth considering because it is single-phase and original-weights friendly; evidence at video scale is limited.

---

## 8. Distributed low-bandwidth training

Current Nivra: DiLoCo with fp16 deltas, Nesterov outer (lr 0.7, momentum 0.9), clipped and trimmed-mean aggregation, streaming on a Pi.

| Method | Key result | Fit for Nivra |
|---|---|---|
| DiLoCo (https://arxiv.org/abs/2311.08105) | H=500, outer Nesterov lr 0.7 / momentum 0.9 robust | Already used. |
| DiLoCo scaling laws (https://arxiv.org/abs/2503.09799, NeurIPS 2025) | Scales predictably to 2.4B; optimal outer lr roughly constant; DiLoCo can beat DP even at small M | Validates the design. |
| Streaming DiLoCo (https://arxiv.org/abs/2501.18512) | Fragment sync lowers peak bandwidth; overlapped communication; **4-bit outer gradients without loss** | **Adopt quantisation and fragments.** |
| INTELLECT-1 (https://arxiv.org/abs/2412.01152) | int8 pseudo-gradients, H≈100-500, fp32 accumulation, dynamic join/leave | Practical lessons for volunteers. |
| SNOO (https://arxiv.org/abs/2510.15830) | DiLoCo's benefit comes mostly from Nesterov on the pseudo-gradient | Confirms keeping Nesterov. |
| DeMo / DisTrO (https://arxiv.org/abs/2411.19870) | 85x less data per step than AdamW-DDP; replaces the optimizer | Higher complexity; not needed when DiLoCo already syncs rarely. |
| SWARM / Hivemind | Pipeline across peers | Not needed; models fit on one GPU. |

Recommendations:
- **H:** inner steps should be tied to wall-clock time, not step count (heterogeneous GPUs). Aim for rounds of about 30-90 minutes; on a 2060 at 256p that is likely H≈100-300. Normalise each contribution by its **number of samples processed** (weighted mean), and keep `clip_factor`/trimmed mean for robustness against bad actors.
- **Compression:** int8 per-tensor (or per-row) absmax deltas now (2x smaller than fp16), then 4-bit with **error feedback** (keep the local quantisation residual and add it to the next round's delta). Accumulate in fp32 on the aggregator (the Pi does this streaming).
- **Stragglers:** accept late deltas computed against version v-1 with a staleness discount (async-DiLoCo style), rather than dropping them. Under heavy staleness, outer momentum can hurt, so cap the staleness at 1.
- **Outer lr warm-up:** start outer lr lower (0.3-0.5) for the first few rounds after a stage change, or when the number of workers changes a lot.
- Keep the deterministic data assignment per worker so deltas are not computed on duplicated data.

---

## 9. Long video

| Method | Idea | Source |
|---|---|---|
| FramePack | Compress older frame context to fixed length; anti-drift sampling; 13B on 6 GB VRAM (but ~20 GB+ system RAM; RTX 20xx not officially tested) | https://github.com/lllyasviel/FramePack, https://arxiv.org/abs/2504.12626 |
| Diffusion Forcing | Per-token independent noise levels; autoregressive rollouts with clean history | https://arxiv.org/abs/2407.01392 |
| SkyReels-V2 | Diffusion-forcing fine-tune of Wan for infinite length; 1.3B variant exists | https://arxiv.org/abs/2504.13074 |
| CausVid / Self Forcing / Self-Forcing++ | Distil a bidirectional teacher into a causal few-step student with a KV cache; train on own rollouts; 4-minute videos | https://arxiv.org/abs/2412.07772, https://arxiv.org/abs/2506.08009, https://arxiv.org/abs/2510.02283 |
| MAGI-1 | Chunk-wise autoregressive denoising, monotonically increasing noise by chunk | https://arxiv.org/abs/2505.13211 |
| FIFO-Diffusion | Training-free diagonal denoising queue | https://arxiv.org/abs/2405.11473 |
| SANA-Video | Block-linear attention with constant-memory state for minute-long video | https://arxiv.org/abs/2509.24695 |

**Is draft, then SDEdit refine with anchored boundaries, then stitch reasonable?** Yes, as a training-free first version. It resembles hierarchical keyframe-then-infill schemes and needs no extra training. Its weaknesses: (a) SDEdit at high strength drifts away from the draft; at low strength it cannot add detail; (b) boundary anchoring by latent replacement produces visible seams when the 8x-temporal latent frame straddles a cut point; (c) global identity drift is limited only by the draft's quality.

Improvements, cheapest first:
1. **Overlapping windows with linear latent blending** across 1-2 latent frames (already partly done through anchored boundaries). Blend predicted velocities, not just final latents, over the last ~30% of steps.
2. **Train with diffusion-forcing noise on part of the batch** (#10). Then the refine step can condition on *clean* neighbour latents at t=0 while denoising the segment, instead of relying on replacement anchors. This is the main seam fix and costs nearly nothing in training.
3. Add **first-frame/last-frame conditioning** (an I2V-style mask channel, as in Open-Sora 2.0's 2k+1 channels) so segments can be generated from given boundary frames. This also yields image-to-video for free.
4. Longer term: a causal student (Self Forcing) for streaming, once a strong base exists.

---

## 10. Evaluation without large compute

- **Primary: held-out flow-matching loss** at fixed t-bins (already present in `eval_job.py` per_t). SD3 showed validation loss correlates strongly with human preference and GenEval. It is cheap and deterministic, and well suited to DiLoCo round-by-round monitoring.
- **VAE:** PSNR/SSIM plus **edge F1 / line-art preservation** on a stylized held-out set; temporal flicker (warped-frame error).
- **Generation, small budget:**
  - Prompt adherence: a fixed stylized prompt suite (about 100 prompts) judged by a VLM (Qwen3-VL, Apache-2.0) with yes/no questions per prompt (VQAScore/DSG-style). Nivra already has `evaluation/adherence.py` and `stylized_prompts.json`.
  - Temporal consistency: DINOv2 cosine similarity between frames (VBench "subject consistency" uses DINO; Apache-2.0).
  - Motion: mean optical-flow magnitude (RAFT is BSD-3) to catch frozen videos ("dynamic degree").
  - Distribution: **JEDi** (https://arxiv.org/abs/2410.05203) needs ~16% of FVD's samples, but V-JEPA weights are CC-BY-NC (verify; V-JEPA 2 may differ). For internal research use that is fine; do not ship it. FVD's I3D weights are Apache-2.0 (Kinetics data terms aside) but need thousands of samples.
- VBench (https://github.com/Vchitect/VBench) dimensions use mixed licences (CLIP, LAION aesthetic, AMT, RAFT, GRiT). Use individual dimensions, not the whole suite.

---

## Change list mapped to Nivra components

| # | Component / file | Change | Effort | Risk |
|---|---|---|---|---|
| 1 | `ai/nivra/compression/hc_vae.py` (`vae_losses`) + a new `discriminator.py` | Patch GAN (3D or 2+1D), hinge loss, adaptive weight, start after N steps; set `edge_weight` 0.1-0.5 for stylized data | M | GAN instability; mitigate with late start, low weight, R1 penalty |
| 2 | VAE training script / `hd_pipeline.py` | Stage plan: 2D image AE, then inflated causal 3D at low resolution and short clips, then target fine-tune; image:video ~1:4 mix | S-M | None significant |
| 3 | `HCVAEConfig.temporal_down`, `ai/configs/*` | Train and compare 4x-temporal (`[2,3]`) against 8x on a held-out stylized set; choose before locking | M | 2x tokens on time axis if 4x wins |
| 4 | `hc_vae.py` | DC-AE-1.5 structured latent (random trailing-channel masking during AE training) | S | Small reconstruction cost |
| 5 | `hd_pipeline.py`, `latent_shard.py`, `encode_job.py`, VAE config | Per-channel `latent_shift`/`latent_scale` vectors `[C]` | S | Shard schema bump (`nivra.*/2`); old shards invalid |
| 6 | `models/dit.py`, `training/hd_trainer.py`, `distributed/hd/train_job.py` | SRA loss (EMA teacher, projection head, layer pair such as 4 to 8 of 24), optional REPA with precomputed DINOv2 features in shards | M | Extra step time (~20-30%); tune weight (REPA used 0.5) |
| 7 | `training/hd_pipeline.py` stages, `ai/configs` | Image-first stage (T=1), joint image+video stage | S | None |
| 8 | `ai/nivra/captioning/models.py`, `runner.py` | Qwen3-VL-2B/4B backend, structured JSON prompt, short and dense captions, no ASR text in visual caption | M | Captioning speed on a 2060; licence verification |
| 9 | `ai/nivra/text/*` | umT5-base licensed add-on (cached embeddings); optional broad-corpus pretraining of NivraText | S-M | Encoder choice must be fixed before DiT training |
| 10 | `ai/nivra/data/scene.py`, ingest | Hold detection, drawing-rate resampling or `effective_fps` conditioning, OCR/subtitle filter, motion and blur filters | S-M | Threshold tuning per source |
| 11 | `models/dit.py` (`rf_loss`), `hd_trainer.py` | Diffusion-forcing per-latent-frame t on ~25% of batches; first/last-frame mask conditioning channel | S-M | Slightly slower convergence on the main task; keep the fraction modest |
| 12 | `distributed/hd/update.py`, `aggregate.py` | int8 then 4-bit deltas with error feedback; sample-weighted mean; staleness-1 acceptance; outer-lr warm-up | M | Error-feedback state per worker must persist across rounds |
| 13 | `models/dit.py`, trainers | fp32 islands (adaLN, t-embed, norms, softmax, loss); GradScaler; optional sandwich norm / registers | S | None |
| 14 | `ai/nivra/inference/*` | TeaCache-style step skipping; later rCM/DMD2 or shortcut distillation | M-L | Distillation needs a converged teacher |
| 15 | `ai/nivra/evaluation/*` | Per-t held-out loss dashboards, VLM-judged adherence, DINOv2 consistency, flow-magnitude motion, VAE edge F1 | S | Licence of each judge model |
| 16 | `models/dit.py` (720p only) | Block-linear or windowed attention for the high-resolution stage | L | Architecture change mid-curriculum; decide before stage D |
| 17 | Docs (`HD_TRAINING.md`) | State compute realities: base 256p is reachable alone; medium/large need many volunteers | S | — |

### Suggested order before the first real training run
1. Finalise the VAE (changes 1-5) on a 256p stylized set and **lock it**. Every later change depends on its latents.
2. Choose and freeze the text-encoder track (change 9), recaption with Qwen3-VL (change 8), apply data filtering and hold handling (change 10).
3. Run a single-GPU base (213M) image-then-video run at 256p with SRA (6, 7), diffusion forcing (11) and fp32 islands (13). Compare held-out loss curves with and without SRA for ~10k steps before committing volunteers.
4. Then turn on volunteer DiLoCo with int8 deltas (12).

### Uncertainty and licence flags
- Hardware throughput ratios are order-of-magnitude estimates, not measurements on the owner's card. Measure steps/s on the 2060 for the base config at 256p before planning.
- Qwen3-VL, umT5 and DINOv2 licences are stated as Apache-2.0 by model cards or aggregators. Confirm the LICENSE file in each official repo.
- V-JEPA (for JEDi) is believed to be CC-BY-NC: internal evaluation only.
- Gemma-family models (T5Gemma, Gemma-2) carry use-restriction terms; avoid them for a clean licence story.
- Qwen2.5-VL-3B is under the Qwen research (non-commercial) licence.
- Training-data copyright is the largest legal risk. It is not addressed by any model licence.
- Register-token, SRA and shortcut-model results are recent and mostly image-domain. Treat their video benefit as plausible, not proven. Run small ablations first.
