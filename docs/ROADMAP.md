# Roadmap

Milestones follow the master specification (Part 36). Status values: **Verified complete**, **Implemented but not verified**, **Experimental**, **Pending**, **Blocked**. Evidence lives in `docs/TEST_RESULTS.md`, `docs/PERFORMANCE.md` and `docs/research/EXPERIMENTS.md`.

| Milestone | Scope | Status |
|-----------|-------|--------|
| A Repository and installers | monorepo, `install.sh/.bat`, `start.sh/.bat`, `nivra doctor/repair` | Implemented; Linux controller install verified on the Pi; Windows installer not yet run end-to-end |
| B Controller | Pi service, PostgreSQL, Studio, project codes | Controller + DB verified on the Pi; Studio UI in progress |
| C Worker | cross-platform worker, enrollment, hardware, admin controls | Verified (Windows worker ↔ Pi controller; e2e test) |
| D Distributed jobs | queue, execution, verified results, contributions, retry/recovery | Verified (tests + physical Pi/laptop run + load test) |
| E Small original AI | tokenizer/VAE, model, training, checkpoints, first video | In progress (VAE trained; generator training) |
| F Multi-worker training | DiLoCo rounds, accepted updates, versioning, recovery | Implemented; physical run pending |
| G Video quality research | prompt fidelity, consistency, resolution, duration | Pending (baseline benchmark first) |
| H ComfyUI | custom nodes, workflows, model loading, export | Nodes implemented; ComfyUI run pending |
| I Audio | speech, effects, timeline, lip-sync | Pending |
| J Release preparation | README, model cards, security review, benchmarks | In progress |

## Next research steps (after the 64×64 baseline)
1. Initialise distributed runs from single-node checkpoints (bundle import) instead of random init.
2. Larger latent model on the RTX 2060 (FP16 autocast, activation checkpointing) — requires running a worker on that machine.
3. Learned text conditioning beyond the closed synthetic vocabulary (original encoder trained on licensed captions).
4. Longer clips via chained continuation with drift measurement.
5. Audio: event timeline synthesis aligned to detected motion events, then learned models.
