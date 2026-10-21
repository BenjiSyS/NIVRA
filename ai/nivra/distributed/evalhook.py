"""Optional held-out loss before/after an aggregation (worker-grade dependencies; lazy import)."""
from __future__ import annotations

from ..config import build_gen, load_config, synth_config
from ..modelio import load_vae
from ..training.tasks import encode_dataset, heldout_flow_loss
from . import version as ver

HELDOUT_SEED = 424242
HELDOUT_START = 2_000_000
HELDOUT_N = 16


def heldout_before_after(base_manifest, base_weights, new_weights, vae_path) -> tuple:
    from safetensors.torch import load_file
    m = ver.read_manifest(base_manifest)
    cfg = load_config(m["config"])
    vae = load_vae(vae_path)
    lat, ids = encode_dataset(vae, synth_config(cfg), HELDOUT_SEED, HELDOUT_START, HELDOUT_N)
    out = []
    for p in (base_weights, new_weights):
        model = build_gen(cfg)
        model.load_state_dict(load_file(str(p)))
        out.append(heldout_flow_loss(model, lat, ids[:, 0]))
    return out[0], out[1]
