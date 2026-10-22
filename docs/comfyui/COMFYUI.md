# Nivra in ComfyUI

The custom node package lives in `integrations/comfyui/nivra_comfyui`. It exposes only features the
Nivra model really implements:

| Node | Inputs → Outputs |
|------|------------------|
| **Nivra Remote Text-to-Video (network)** | controller URL, prompt, seed, steps, cfg_scale, upscale, timeout, optional project/model → `IMAGE` frames, MP4 path. Runs on a worker computer of your Nivra network, not on the ComfyUI machine |
| **Nivra Model Loader** | bundle (folder in `models/nivra/`), device (`auto`, `cpu`, `cuda:N`) → `NIVRA_MODEL` |
| **Nivra Text-to-Video** | model, prompt, seed, steps, cfg_scale, decode_tile → `IMAGE` frames, `NIVRA_STATE` |
| **Nivra Image-to-Video** | model, image, prompt, seed, steps, cfg_scale → `IMAGE`, `NIVRA_STATE` |
| **Nivra Video Continuation** | model, state, known_latent_frames, prompt (optional) → `IMAGE` (whole video so far), `NIVRA_STATE` |
| **Nivra Export MP4** | frames, fps, upscale, filename_prefix → MP4 in ComfyUI's `output/` folder |

Not provided (not implemented by the model yet): negative prompts, external VAEs, audio nodes,
reference-identity conditioning. Nivra latents are only decodable by the Nivra VAE that ships in the
same bundle; the loader refuses a VAE whose SHA-256 differs from the one the model was trained with.

## Two ways to run

* **Local nodes** (Model Loader → Text-to-Video → Export): the model runs inside ComfyUI on this computer.
  Needs a model bundle in `ComfyUI/models/nivra/`.
* **Nivra Remote Text-to-Video**: ComfyUI only sends the prompt; the Nivra controller queues it and any
  connected worker computer (for example a GPU PC) renders it, then the frames come back into ComfyUI.
  Many people — each with their own ComfyUI, or sharing one — can use the same network at the same
  time; the controller's queue spreads their requests over all idle AI workers. The ComfyUI machine
  needs no model files and no GPU.

### Setting up the remote node (each person)

1. Ask the Nivra administrator for an invitation (a **member** account is enough to make videos)
   (`bin/nivra admin invite` on the controller, or Studio → People).
2. Sign in to Nivra Studio → **Settings → API tokens → Create token**, copy it once.
3. Give it to ComfyUI **outside the workflow** (so shared workflows never contain it), either:
   * environment variable `NIVRA_TOKEN=nivra_pat_...` before starting ComfyUI, or
   * a text file `ComfyUI/user/nivra_token.txt` containing only the token.
4. In the node, set `controller_url` to the Studio address (e.g. `https://studio.example.com`) and
   type a prompt. Load `examples/workflows/nivra_remote_network_api.json` for a ready-made graph.
5. Revoke a token any time in Studio Settings; it stops working immediately.

## Installation (manual Git install)

1. Install ComfyUI following its official instructions and confirm it starts (`python main.py`).
2. Use the same Python that runs ComfyUI for every command below. Check it:
   `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
3. Get Nivra: `git clone <nivra repository URL>` (or extract a release package).
4. Install the node package: copy `integrations/comfyui/nivra_comfyui` into `ComfyUI/custom_nodes/`.
5. Install dependencies into ComfyUI's Python:
   `python -m pip install -e <nivra>/ai` (installs the `nivra` model package; torch is reused).
   Alternative without pip: set the environment variable `NIVRA_AI_DIR=<nivra>/ai` before starting ComfyUI.
6. Download/copy a model bundle (a folder with `weights.safetensors`, `model_version.json`,
   `vae.safetensors`) into `ComfyUI/models/nivra/<bundle-name>/`. Bundles are produced by Nivra
   training runs (see docs/inference/INFERENCE.md); no bundle is published online yet.
7. The tokenizer/"native VAE" needs no separate configuration: the tokenizer is built into the
   `nivra` package and the VAE is inside the bundle.
8. Restart ComfyUI. Search the node menu for "Nivra".
9. Import a workflow from `examples/workflows/` (drag the JSON onto the canvas).
10. In **Nivra Model Loader** choose the bundle and device (`cpu` works everywhere; `cuda:0` for NVIDIA).
11. Type an ordinary prompt, e.g. `two blue squares move up` or `en röd cirkel rör sig från vänster till höger`.
    The current research model only knows its synthetic-scene vocabulary (shapes, colours, counts,
    directions in English and Swedish); unknown words are ignored and reported in the console.
12. Sampling: `steps` 20–30 and `cfg_scale` ~3 are the defaults that were evaluated.
13. Queue the prompt. The MP4 path is shown by **Nivra Export MP4** and saved in `ComfyUI/output/`.

ComfyUI Manager: the package is **not registered** in the ComfyUI registry yet, so use the manual
install above.

## Troubleshooting

* *Missing nodes*: the folder must be `custom_nodes/nivra_comfyui/` containing `__init__.py`; check the
  ComfyUI console for `ImportError: No module named nivra` → step 5.
* *GPU out of memory*: choose `cpu`, or set `decode_tile` to 4–8 (tiled VAE decode).
* *Invalid model format*: "Invalid or incompatible Nivra bundle" means a missing file, a checksum
  mismatch, a different architecture/tokenizer version, or a foreign VAE.
* *Version compatibility*: the bundle's `model_version.json` records `architecture_version` and
  `tokenizer_version`; the node checks both against the installed `nivra` package.
