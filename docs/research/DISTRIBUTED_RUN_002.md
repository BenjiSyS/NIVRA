# Distributed training run 002 — continuing the trained model on two workers

Date: 2026-10-21. Controller on Raspberry Pi 5 (system torch 2.6.0, aarch64), workers `benji-laptop`
(Windows, torch 2.14.1 CPU) and `benjipi-worker` (Pi), both marked **trusted** (required for
model-training jobs since the security hardening).

| Item | Value |
|------|-------|
| Run | `diloco-from-6000` (`8f6cf913-48bc-4699-af02-a494c5adc1b5`) |
| Initialisation | imported bundle `nivra-small-6000` (EMA weights after 6000 single-node steps) via `init_weights_artifact` / `init_manifest_artifact` |
| Inner loop | H = 20 AdamW steps, batch 4, lr 3e-5, disjoint 64-clip shards (`data_seed` 21) |
| Outer | Nesterov SGD lr 0.7, momentum 0.9; new norm bounds (`max_rel_norm` 0.5, per-tensor bound, zero-update rejection, median clipping); held-out gate 10 % |

| Round | Held-out loss before → after (controller-measured) | Accepted / rejected |
|------:|---------------------------------------------------|--------------------:|
| 0 | 0.2903 → 0.2922 | 2 / 0 |
| 1 | 0.2922 → 0.2931 | 2 / 0 |

Findings:
* All controller-side jobs (`ai.validate_update`, `ai.aggregate` with the new bounds) ran correctly on the
  Pi's torch 2.6 — this closes the "untested on torch 2.6/aarch64" gap reported by the AI package work.
* Continuing an already converged model with DiLoCo on tiny shards slightly **worsens** held-out loss
  (+1.0 % over two rounds), matching the single-machine simulation in EXPERIMENTS.md. The increase stayed
  below the 10 % gate, so versions were accepted; the project's active generation model remains the
  imported `nivra-small-6000` bundle unless an operator changes it.
* Conclusion: DiLoCo is useful here for early training (run 001: 2.04 → 1.46), not for fine-tuning this
  converged baseline; larger shards, more inner steps and lower outer learning rates are the next things to test.
