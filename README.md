# Nivra

Nivra is an open-source research project for original generative video: a distributed training
platform (a controller on a Raspberry Pi, volunteer workers on ordinary computers) and an original,
from-scratch text-to-video model family trained on it. Source code is MIT licensed.

> **Honest status:** the platform (controller, workers, scheduler, distributed training orchestration,
> installers) works and is tested. The model is an early *research baseline*: a small model trained
> from random initialisation on procedurally generated videos of coloured shapes at 64×64, 16 frames.
> It is **not** a general-purpose video generator yet. The long-term 1280×720 / 240-frame targets
> are research goals, not current capabilities. See [Current Project Status](#2-current-project-status).

## Contents

1. [Introduction](#1-introduction)
2. [Current Project Status](#2-current-project-status)
3. [Features](#3-features)
4. [Actual Video Examples](#4-actual-video-examples)
5. [System Architecture](#5-system-architecture)
6. [Supported Hardware](#6-supported-hardware)
7. [Quick Start](#7-quick-start)
8. [Windows Installation](#8-windows-installation)
9. [Linux Installation](#9-linux-installation)
10. [Raspberry Pi Controller](#10-raspberry-pi-controller)
11. [Worker Setup](#11-worker-setup)
12. [Project Enrollment Codes](#12-project-enrollment-codes)
13. [Cloudflare and Domain](#13-cloudflare-and-domain)
14. [Network Architecture](#14-network-architecture)
15. [Distributed Training](#15-distributed-training)
16. [Controller Website](#16-controller-website)
17. [Worker Website](#17-worker-website)
18. [Training Management](#18-training-management)
19. [Checkpoints](#19-checkpoints)
20. [Datasets](#20-datasets)
21. [Model Architecture](#21-model-architecture)
22. [Model Variants](#22-model-variants)
23. [Model Downloads](#23-model-downloads)
24. [Text-to-Video Generation](#24-text-to-video-generation)
25. [Prompt Guide](#25-prompt-guide)
26. [Multilingual Prompts](#26-multilingual-prompts)
27. [Character Consistency](#27-character-consistency)
28. [Video Continuation](#28-video-continuation)
29. [Audio Generation](#29-audio-generation)
30. [ComfyUI Installation](#30-comfyui-installation)
31. [ComfyUI Workflows](#31-comfyui-workflows)
32. [Recommended Settings](#32-recommended-settings)
33. [Low-VRAM Optimization](#33-low-vram-optimization)
34. [Development Setup](#34-development-setup)
35. [Repository Structure](#35-repository-structure)
36. [API Reference](#36-api-reference)
37. [Building from Source](#37-building-from-source)
38. [Running Tests](#38-running-tests)
39. [Benchmarks](#39-benchmarks)
40. [Troubleshooting](#40-troubleshooting)
41. [Security](#41-security)
42. [Contributing](#42-contributing)
43. [Roadmap](#43-roadmap)
44. [License](#44-license)
45. [Credits](#45-credits)

---

## 1. Introduction

Nivra has two halves:

* **Nivra Network** — a controller (Go + PostgreSQL) that runs on a Raspberry Pi 5 and coordinates
  workers over HTTPS; **Nivra Worker** agents that run on Windows/Linux/macOS computers, execute
  verifiable jobs and AI training tasks, and report results; **Nivra Studio**, the administration website.
* **Nivra Models** — an original video model family (tokenizer/VAE + flow-matching video transformer)
  whose weights are trained from scratch by this project. No pretrained video weights are used.

## 2. Current Project Status

The authoritative, evidence-linked status is [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).
Summary:

| Area | Status |
|------|--------|
| Controller, database, enrollment, scheduler, contributions | Verified (unit, database, end-to-end and fault-injection tests; running on the Pi) |
| Workers (Windows/Linux), physical multi-machine jobs | Verified (Windows laptop worker ↔ Raspberry Pi controller) |
| 50-worker control-plane load test on the Pi | See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Nivra Small model (64×64, 16 frames, synthetic domain) | Trained on CPU; see [docs/research/EXPERIMENTS.md](docs/research/EXPERIMENTS.md) |
| Distributed DiLoCo training | Implemented; physical results in EXPERIMENTS.md |
| ComfyUI nodes | Implemented; test status in [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) |
| Audio, speech, lip-sync | Not started |
| 720p / 240-frame generation | Research goal, not implemented |

## 3. Features

* One `nivra` binary: controller, worker, diagnostics (`nivra doctor`), repair, admin tools, load tester.
* Project enrollment codes (`NV-XXXX-XXXX-XXXX-XXXX`, 80 bits) displayed in Studio; rotation and closing.
* Per-device Ed25519 identities, signed short-lived sessions, revocation, controller identity pinning.
* Durable PostgreSQL scheduler with fencing tokens, idempotent results, lease expiry and reassignment.
* Reliable worker commands (pause, resume, stop task, disconnect) with visible acknowledgement states.
* Resumable, checksummed, deduplicated artifact transfers.
* Server-side verified contributions and an opt-in public leaderboard.
* DiLoCo-style distributed training with versioned global models, update validation and checkpoints.
* Live dashboard over Server-Sent Events; persistent searchable logs.
* Original model: 3D-conv VAE tokenizer + factorised space/time transformer trained with rectified flow,
  bilingual (English/Swedish) prompts, video continuation and image-to-video conditioning.
* ComfyUI custom nodes.

## 4. Actual Video Examples

Samples are produced by `nivra.cli generate` from the trained checkpoint and listed with their exact
prompts, seeds and checkpoint ids in [docs/research/EXPERIMENTS.md](docs/research/EXPERIMENTS.md).
They are small (64×64, upscaled ×4 for viewing) synthetic-domain clips. No example in this
repository was produced by any other model.

## 5. System Architecture

```
 Browser (Nivra Studio)          Worker computers (Nivra Worker)
        │  HTTPS                      │  HTTPS (device-key sessions)
        ▼                             ▼
  Cloudflare edge ── tunnel ── cloudflared on Raspberry Pi
                                      │ http://127.0.0.1:8740
                               nivra controller (Go)
                     ┌────────────────┼──────────────────┐
              PostgreSQL 17     artifact store      Python (torch) for
             127.0.0.1:5433     (content-addressed)  update validation +
                                                     DiLoCo aggregation
```

Details: [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md),
[docs/architecture/EXECUTOR_PROTOCOL.md](docs/architecture/EXECUTOR_PROTOCOL.md).

## 6. Supported Hardware

| Role | Tested | Notes |
|------|--------|-------|
| Controller | Raspberry Pi 5 16 GB, Debian 13 arm64 | Any 64-bit Linux with Docker or PostgreSQL 13+ |
| Worker (control plane + builtin jobs) | Windows 11 x64 | Linux/macOS binaries are built; Linux worker used on the Pi |
| Worker (AI, CPU) | Intel i5-8250U; Raspberry Pi 5 (torch 2.6) | Slow but functional for Nivra Small |
| Worker (AI, NVIDIA) | Not yet tested | RTX 2060 6 GB is the primary target: FP16 autocast, no bf16 |

Projected GPU profiles are in [docs/inference/INFERENCE.md](docs/inference/INFERENCE.md) and are marked untested.

## 7. Quick Start

See [QUICKSTART.md](QUICKSTART.md) — where everything lives, start/stop, adding workers.

## 8. Windows Installation

### One file: `Nivra-Setup.bat` (recommended for other people)

Send someone only [`Nivra-Setup.bat`](Nivra-Setup.bat). Double-clicking it:

1. installs Git with winget if it is missing (it asks first),
2. clones Nivra into `%USERPROFILE%\Nivra` (or the folder given as the first argument),
3. installs missing tools with winget (Go and Node.js to build, Python 3.12 to run the model),
4. detects the hardware: an NVIDIA GPU gets CUDA PyTorch, matched to the GPU generation; anything
   else gets CPU PyTorch,
5. builds `Nivra.exe` in the Nivra folder and adds Desktop and Start menu shortcuts.

Running it again updates Nivra (`git pull`) and rebuilds. `NIVRA_SETUP_YES=1` answers every question
with yes. Nothing starts automatically at boot.

**The Nivra App (`Nivra.exe`)** opens a window where you type a prompt and the video comes back as the
reply, like a chat. It is not a chatbot: the text goes to the video model. The first time, enter the
controller URL and a personal API token (Studio → Settings → API tokens). Generation runs on whichever
worker computer the controller picks (one controller, many workers), so a laptop without a GPU can
still use a GPU computer elsewhere. The same window has **This computer as a worker**: enroll it with a
project code and it trains and generates for everyone too. It only gets jobs that fit its RAM/VRAM
budget, so HD jobs that need a GPU never land on a CPU-only computer. Closing the window or
**Quit Nivra** stops the app and this computer's worker. Details: [docs/getting-started/APP_UI.md](docs/getting-started/APP_UI.md).

### Manual

1. Install Git (optional) and download/clone Nivra.
2. Double-click `install.bat` and choose **2 — Worker** (or **1 — Controller** if you have PostgreSQL).
3. Answer whether to enable AI jobs (installs CPU or CUDA PyTorch into `%LOCALAPPDATA%\Nivra\venv`).
4. Run `start.bat`; the Worker website opens at `http://127.0.0.1:8741/`.

## 9. Linux Installation

```bash
bash install.sh            # menu: 1 Controller, 2 Worker, 3 Repair, 4 Diagnostics, 5 Exit
bash start.sh worker       # or: bash start.sh controller
```

Non-interactive: `bash install.sh --role worker --with-ai --yes`.

## 10. Raspberry Pi Controller

`bash install.sh --role controller` creates a dedicated PostgreSQL container `nivra-pg` bound to
`127.0.0.1:5433` (it never touches other databases), a Python environment that reuses the system
PyTorch, `config/controller.json`, secrets and the database schema, then the first project.
Runtime data lives in `~/NivraData`. Nothing starts automatically at boot unless you install the
opt-in units in [deploy/systemd](deploy/systemd).

## 11. Worker Setup

Workers need the controller URL and the project code. They report CPU, RAM, NVIDIA GPUs (via
`nvidia-smi`) and whether the Python AI executor works. Resource limits (CPU threads, GPU, RAM/VRAM
budget, disk cache, bandwidth, operating hours) are set in the Worker website.

## 12. Project Enrollment Codes

* Shown in the top-right of Nivra Studio (operators and admins), with Copy and Manage.
* Only used to enroll; each device then authenticates with its own key. Rotating a code does not
  disconnect enrolled devices. Closing enrollment stops new devices.
* Rate limited (10 attempts/minute per IP); failures are audited.

## 13. Cloudflare and Domain

Planned hostnames: `studio.benjiserver.com` (admin) and `api.benjiserver.com` (workers/API) via the
existing Cloudflare Tunnel — step-by-step in [deploy/cloudflare/README.md](deploy/cloudflare/README.md).
Public routes are **not** enabled yet; when the controller is stopped, the sites are offline.

## 14. Network Architecture

Three logical channels: control (sessions, heartbeats every 15 s, job leases, commands), telemetry
(metrics in heartbeats, batched logs) and artifacts (chunked resumable uploads, Range downloads).
Protocol: [docs/api/API.md](docs/api/API.md).

## 15. Distributed Training

Two different capabilities are measured separately:

1. **Distributed task execution** — verified jobs across machines (builtin verifiable jobs, dataset shards).
2. **Distributed neural-network optimisation** — DiLoCo rounds: each worker trains H steps from the
   current global version on its own data shard and uploads a pseudo-gradient; the controller validates
   updates (base version, shapes, finiteness, norm bounds) and applies an outer Nesterov step to create
   the next version. Stale and duplicate updates are rejected. See [docs/training/DISTRIBUTED.md](docs/training/DISTRIBUTED.md).

## 16. Controller Website

Nivra Studio pages: Overview, Workers, Queue, Training, Experiments, Checkpoints, Models, Generate,
Datasets, Contributions, Logs, Settings. Live updates via SSE; pause/resume/stop/revoke actions call
the real API and show "requested" vs "confirmed".

## 17. Worker Website

Connect with a project code; then Overview, Current task, Hardware, Resource settings, Contributions,
Logs and Settings, plus Start / Pause / Resume / Stop / Change Project.

## 18. Training Management

Create a run (architecture, rounds, workers per round, inner steps, learning rates, VAE artifact,
optional bundle to start from), then pause, resume, stop at a safe boundary, save or load checkpoints.
Runs continue independently of the browser and survive controller restarts.

## 19. Checkpoints

Single-node training checkpoints: safetensors + JSON manifests with SHA-256 per file, atomic writes,
exact resume (model, EMA, optimizer, scheduler, RNG, sampler position). Distributed checkpoints: every
aggregated global version, with retention that never deletes protected, initial, best or latest ones.

## 20. Datasets

Current training data is procedurally generated (legal, original): 1–3 coloured shapes with exact
scene specifications and English/Swedish captions, deterministic per (generator version, seed, index).
Workers can generate verified shards (`ai.dataset_shard`), which the controller re-computes.

## 21. Model Architecture

* **Nivra tokenizer**: 3D-conv VAE, ×4 spatial and ×2 temporal compression, continuous latents.
* **Text**: original word-level bilingual tokenizer + small transformer encoder trained jointly.
* **Generator**: rectified-flow transformer on latents with factorised spatial/temporal attention,
  cross-attention to text and adaLN-Zero timestep conditioning.
Details and measured parameter counts: [docs/training/TRAINING.md](docs/training/TRAINING.md).

## 22. Model Variants

| Variant | Status |
|---------|--------|
| Nivra Small (64×64×16, synthetic domain) | Trained (CPU baseline) |
| Nivra Medium | Planned (needs GPU workers) |
| Nivra Large (~15 GB target) | Research goal; not scientifically justified yet |

## 23. Model Downloads

No weights are published yet (publishing requires the owner's authorisation and a model card).
Bundles are produced locally by training runs.

## 24. Text-to-Video Generation

Generation (using a trained model, no training) can run **on the computer you are using** or **on
another computer connected to your Nivra controller** — for example: you type a prompt on a laptop or
phone, the Raspberry Pi controller queues it, and a worker PC (ideally the one with a GPU) renders the
video and sends the MP4 back.

### A. Use a model on another computer (through the controller)

One-time setup:

1. **Controller** running (see [Raspberry Pi Controller](#10-raspberry-pi-controller)).
2. **Worker computer**: install with AI enabled (`install.bat` → 2 — Worker → answer `y` to AI jobs, or
   `bash install.sh --role worker --with-ai`), start it (`start.bat` / `bash start.sh worker`), and enter
   the project code. Its Hardware tab must say "AI executor ready".
3. **A model**: in Nivra Studio → **Models** → **Import bundle**, choose the three files of a trained
   bundle (`weights.safetensors`, `model_version.json`, `vae.safetensors`) and keep "Use as the default
   model for generation" ticked. (On the controller host the same is
   `bin/nivra model import --project Nivra --bundle path/to/bundle`.) Models produced by distributed
   training runs appear here automatically.

Then generate from any computer:

* **Browser:** open Nivra Studio → **Generate**, type a prompt, press Generate. Progress is live; the
  video plays in the page and can be downloaded. Any idle worker with the AI executor picks the job up
  and downloads the model automatically the first time (cached afterwards).
* **Command line** (no install needed except the `nivra` binary):

  ```bash
  nivra generate --url https://studio.benjiserver.com --user alice --prompt "two blue squares move up" --out clip.mp4
  ```

  The password is asked for (or read from `NIVRA_PASSWORD`). Accounts need the **operator** role:
  `bin/nivra admin create --username alice --role operator` on the controller.

### B. Use a model on this computer only (no controller)

After a worker install with AI enabled:

```bash
nivra generate --bundle path/to/model-bundle --prompt "en röd cirkel rör sig från vänster till höger" --out clip.mp4
```

(The same works with Python directly: `python -m nivra.cli generate --bundle DIR --prompt "..." --out DIR`.)

### Which computer does the work?

The controller sends each generation to an idle connected worker whose AI executor works and whose
RAM/VRAM budget fits the job. To force generation onto one machine, pause the others in Studio or
lower their budgets. Generated videos are stored on the controller and listed on the Generate page.

More details and measured timings: [docs/inference/INFERENCE.md](docs/inference/INFERENCE.md).

## 25. Prompt Guide

The current model understands its synthetic vocabulary: counts (one–three), colours (red, green, blue,
yellow, white, purple), shapes (circle, square, triangle) and motions (left to right, right to left,
up, down, stationary). Unknown words are ignored and reported. Prompt-adherence scores are measured,
not guaranteed.

## 26. Multilingual Prompts

English and Swedish are explicit targets, e.g. `två blå kvadrater rör sig uppåt`.

## 27. Character Consistency

Measured for the synthetic domain as colour/size stability of tracked objects across frames
(temporal-consistency metric in the benchmark). Faces and clothing are out of scope for this model.

## 28. Video Continuation

`GenerationState` stores text conditioning, the last latent frames and the temporal position; the next
segment re-imposes those latent frames during sampling (inpainting-style), so continuations are
conditioned on the previous segment rather than concatenated.

## 29. Audio Generation

Not implemented yet. Planned first step: an event timeline derived from detected motion events driving
synthesised sound effects; speech and lip-sync later. No voice cloning.

## 30. ComfyUI Installation

Step-by-step beginner guide: [docs/comfyui/COMFYUI.md](docs/comfyui/COMFYUI.md).

### Shared use from ComfyUI (many people, one network)

Add the **Nivra Remote Text-to-Video (network)** node: ComfyUI sends the prompt to your Nivra controller
and a connected worker computer does the rendering, so several people (each with their own ComfyUI,
no GPU needed) can generate at the same time on the same set of workers. Each person creates a personal
API token in Nivra Studio → Settings → API tokens and sets it as `NIVRA_TOKEN` (or saves it in
`ComfyUI/user/nivra_token.txt`) — never inside a shared workflow. Details: [docs/comfyui/COMFYUI.md](docs/comfyui/COMFYUI.md#two-ways-to-run).

The command line works the same way with a token: `NIVRA_TOKEN=... nivra generate --url https://studio.benjiserver.com --prompt "..."`.

## 31. ComfyUI Workflows

Example workflows: [examples/workflows](examples/workflows).

## 32. Recommended Settings

Sampling steps 20–30, CFG scale 3, seed fixed for reproducibility. Workers: leave half the CPU threads
free on a computer you also use.

## 33. Low-VRAM Optimization

FP16 autocast on CUDA (never bf16, compatible with RTX 20-series), tiled VAE decoding, CPU fallback,
memory-aware RAM/VRAM budgets that the scheduler respects.

## 34. Development Setup

[docs/getting-started/DEVELOPMENT.md](docs/getting-started/DEVELOPMENT.md).

## 35. Repository Structure

See the layout in [docs/getting-started/DEVELOPMENT.md](docs/getting-started/DEVELOPMENT.md#repository-layout).

## 36. API Reference

[docs/api/API.md](docs/api/API.md).

## 37. Building from Source

```bash
bash scripts/build.sh          # web + host binary into bin/
bash scripts/build.sh --all    # cross-compiled release binaries + SHA256SUMS in dist/bin/
```

## 38. Running Tests

```bash
export NIVRA_TEST_DATABASE_URL=postgres://.../nivra_test   # database name must end in _test
go test -p 1 ./...
cd ai && python -m pytest -q tests
cd web && npm run build
```

Latest results: [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md).

## 39. Benchmarks

Control-plane performance: [docs/PERFORMANCE.md](docs/PERFORMANCE.md). Model quality and prompt
adherence (trained vs untrained baseline): [docs/research/EXPERIMENTS.md](docs/research/EXPERIMENTS.md).

## 40. Troubleshooting

`nivra doctor`, then [docs/troubleshooting/TROUBLESHOOTING.md](docs/troubleshooting/TROUBLESHOOTING.md).

## 41. Security

[SECURITY.md](SECURITY.md) and [docs/security/THREAT_MODEL.md](docs/security/THREAT_MODEL.md).

## 42. Contributing

[CONTRIBUTING.md](CONTRIBUTING.md).

## 43. Roadmap

[docs/ROADMAP.md](docs/ROADMAP.md).

## 44. License

Source code: MIT ([LICENSE](LICENSE)). Model weights and datasets will carry their own licenses in their
model cards; the MIT license does not automatically apply to them.

## 45. Credits

Nivra contributors. Built with Go, PostgreSQL, PyTorch, safetensors, React and Vite. Research
references are listed in [docs/research/PRIOR_ART.md](docs/research/PRIOR_ART.md).
