# Nivra Small - prior art and baseline choices

Scope: a short, honest comparison of the approaches that shaped the design of Nivra Small
(`ai/nivra`). References are given by title / authors / year from memory of the literature;
no numbers from those papers are quoted or relied upon. Nothing here is a claim about how
Nivra Small compares to those systems - Nivra Small is a tiny research model trained on a
procedural toy domain (see `EXPERIMENTS.md` for what was actually measured).

## 1. Families of video generators

### 1.1 Video diffusion
Denoising diffusion (Ho, Jain, Abbeel 2020, "Denoising Diffusion Probabilistic Models";
Song et al. 2021, "Score-Based Generative Modeling through SDEs") was extended to video by
"Video Diffusion Models" (Ho et al. 2022), "Imagen Video" (Ho et al. 2022), "Make-A-Video"
(Singer et al. 2022), "Align Your Latents" (Blattmann et al. 2023) and "Stable Video Diffusion"
(Blattmann et al. 2023). Most production systems run diffusion in the latent space of a
pretrained autoencoder ("High-Resolution Image Synthesis with Latent Diffusion Models",
Rombach et al. 2022), which is what makes training on small hardware plausible at all.
Classifier-free guidance ("Classifier-Free Diffusion Guidance", Ho & Salimans 2022) is the
standard way to get prompt adherence.
Trade-off: strong quality, but multi-step sampling and (usually) large pretrained components.

### 1.2 Flow matching / rectified flow
"Flow Matching for Generative Modeling" (Lipman et al. 2023), "Flow Straight and Fast:
Rectified Flow" (Liu, Gong, Liu 2023) and "Stochastic Interpolants" (Albergo &
Vanden-Eijnden 2023) train a velocity field along (near-)straight paths between noise and data
with a plain regression loss. "Scaling Rectified Flow Transformers for High-Resolution Image
Synthesis" (Esser et al. 2024) popularised logit-normal timestep sampling for transformer
backbones.
Trade-off: simpler objective than a noise-prediction diffusion schedule, no noise-schedule
tuning, an Euler ODE sampler is already a reasonable baseline. Straight-path training does not
guarantee that few-step sampling is good; this is measured here, not assumed.

### 1.3 Video transformers: full 3D vs factorized attention
ViViT (Arnab et al. 2021) and TimeSformer (Bertasius et al. 2021, "divided space-time
attention") showed factorized spatial/temporal attention is far cheaper than joint attention;
Latte (Ma et al. 2024) applies the factorized idea to latent video diffusion transformers,
while CogVideoX (Yang et al. 2024) and the Sora technical report (OpenAI 2024, spacetime
patches) lean on full 3D attention at scale. DiT ("Scalable Diffusion Models with
Transformers", Peebles & Xie 2023) supplies the adaLN-Zero conditioning block.
Trade-off: full attention over N = T*H*W tokens costs O(N^2); factorized costs
O(T*S^2 + S*T^2) for S spatial tokens per frame, but couples space and time only through
alternating layers. On a CPU box with 512 latent tokens per clip the factorized form is the
only affordable choice.

### 1.4 Autoregressive video
Discrete-token autoregressive models: VQ-VAE (van den Oord et al. 2017), VQGAN (Esser et al.
2021), VideoGPT (Yan et al. 2021), CogVideo (Hong et al. 2022), Phenaki (Villegas et al. 2022),
MaskGIT-style masked decoding (Chang et al. 2022), MAGVIT / MAGVIT-v2 (Yu et al. 2023),
VideoPoet (Kondratyuk et al. 2023). Hybrid forms: "Diffusion Forcing" (Chen et al. 2024)
combines per-token noise levels with next-token prediction.
Trade-off: natural unbounded-length generation and a single token stream shared with text, but
quantisation error and slow sequential decoding; error accumulation over long rollouts.

### 1.5 Recurrent memory and continuation
Transformer-XL (Dai et al. 2019), Compressive Transformer (Rae et al. 2020), Memorizing
Transformers (Wu et al. 2022) and Recurrent Memory Transformer (Bulatov et al. 2022) carry
state across segments. In diffusion video, continuation is usually done either by conditioning
on previous frames (MCVD, Voleti et al. 2022) or by replacement-style inpainting at sampling
time ("RePaint", Lugmayr et al. 2022; the "reconstruction guidance" / replacement scheme in
Ho et al. 2022).
Nivra uses the training-free replacement form: the previous segment's last latent frames are
re-imposed (noised along the flow path) at every Euler step. No learned recurrent memory
exists in Nivra Small; the `GenerationState` is an explicit, serialisable record of text
conditioning + last latent frames + position, not a learned hidden state.

### 1.6 Hierarchical / cascaded latents
Cascaded diffusion (Ho et al. 2022, "Cascaded Diffusion Models for High Fidelity Image
Generation"; Imagen Video's spatial and temporal super-resolution cascade), VQ-VAE-2 (Razavi et
al. 2019) and W.A.L.T. (Gupta et al. 2023) use multi-scale or multi-stage latents.
Not implemented in Nivra Small: it has a single 4x spatial / 2x temporal latent level. A
temporal-hierarchy (keyframes then interpolation) is the obvious next experiment for longer
clips and is listed as future work in `EXPERIMENTS.md`.

### 1.7 Video tokenizers: continuous vs discrete
Continuous KL-regularised autoencoders (Kingma & Welling 2014 VAE; Rombach et al. 2022; the
3D causal VAEs in Open-Sora/CogVideoX-class systems) vs discrete tokenizers (VQ-VAE, VQGAN,
MAGVIT-v2 with lookup-free quantisation, Yu et al. 2023).
Trade-off: continuous latents pair naturally with flow/diffusion generators and avoid codebook
collapse; discrete tokens pair with autoregressive or masked generators. Nivra uses a
continuous 3D-conv VAE with a small KL weight plus an explicit temporal-difference loss.

### 1.8 Low-communication distributed training
Local SGD (Stich 2019, "Local SGD Converges Fast and Communicates Little"; Lin et al. 2020,
"Don't Use Large Mini-Batches, Use Local SGD"), FedAvg (McMahan et al. 2017) and server-side
optimisers (FedOpt, Reddi et al. 2021) are the ancestors of DiLoCo ("DiLoCo: Distributed
Low-Communication Training of Language Models", Douillard et al. 2023): H inner AdamW steps per
worker, then an outer optimiser (Nesterov momentum SGD) applied to the averaged
"pseudo-gradient" (base minus local weights). Open reimplementations include OpenDiLoCo
(Jaghouar et al. 2024). Robust aggregation against faulty workers: coordinate-wise trimmed
mean / median (Yin et al. 2018, "Byzantine-Robust Distributed Learning: Towards Optimal
Statistical Rates") and Krum (Blanchard et al. 2017).
Trade-off: communication once per H steps instead of per step, tolerant of slow links (the
Nivra target is home machines over the internet); but extra staleness, and the behaviour at the
very small scale used here is something we measure rather than extrapolate from LLM results.

## 2. Baseline choices for Nivra Small, and why

| Decision | Choice | Why (and the alternative not taken) |
|---|---|---|
| Generative objective | Rectified flow, v-target = noise - x0, logit-normal t, Euler sampler | Simplest regression loss, no schedule tuning; fits the "train from scratch on CPU" constraint. Not taken: epsilon-prediction diffusion with a tuned schedule. |
| Latent space | Own 3D-conv VAE, 4x spatial / 2x temporal, 4 channels, trained here | Required: no pretrained weights. Latents shrink the transformer to 512 tokens/clip, which is what makes ~0.6 s/step on 4 CPU cores possible. Not taken: pixel-space generation, discrete VQ tokens. |
| Backbone | DiT-style transformer, factorized spatial + temporal self-attention, text cross-attention, adaLN-Zero | Factorized attention is the only affordable form here (section 1.3); adaLN-Zero is the well-tested conditioning scheme. Not taken: full 3D attention, U-Net. |
| Text | Closed bilingual word vocabulary (70 entries), small transformer encoder trained jointly | Domain is a closed lexicon so a pretrained text encoder would add nothing and violate the no-pretrained-weights rule. Out-of-vocabulary words map to UNK (reported). |
| Conditioning dropout | 10% empty caption, CFG at sampling | Standard CFG recipe. |
| Continuation | Replacement-style temporal inpainting at sampling time | Needs no extra training and no new architecture; weakness: the model never learned to use the known frames, so quality depends on how well the single-clip distribution covers the continuation (the synthetic generator was designed so that motion is constant-speed and start positions are spread, making continuations in-distribution). |
| Distributed | DiLoCo-style: inner AdamW, outer Nesterov SGD (lr 0.7, momentum 0.9), trimmed mean for >= 3 updates | Matches the project goal of low-communication contribution from home machines; aggregation uses only torch/numpy/safetensors so it can run on the Raspberry Pi controller. |
| Data | Procedural synthetic scenes with exact ground-truth scene specs | Legal, original, and gives an objective prompt-adherence metric. The price: results say little about natural video. |
| Evaluation | Classical colour/shape blob detector + tracks, validated against ground truth first | Gives interpretable per-error-type counts (missing / wrong colour / wrong direction / invented / count). Limited to the toy domain by construction. |

## 3. What this package deliberately does not claim

* No claim of competitiveness with any published model.
* No natural-video capability; the dataset is a closed procedural domain.
* GPU (CUDA/FP16) code paths exist (autocast fp16 + GradScaler, `F.scaled_dot_product_attention`)
  but were never executed on a GPU - see `docs/training/TRAINING.md`.
