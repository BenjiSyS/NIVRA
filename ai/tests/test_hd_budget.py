"""The HD budget tool runs end to end (quick mode) and keeps measured and estimated numbers apart."""
import importlib.util
from pathlib import Path

import torch

from nivra.models.dit import DiTConfig, DiTBlock
from nivra.models.rope import rope_3d

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "hd_budget.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("hd_budget", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flop_model_matches_counter():
    tool = load_tool()
    from torch.utils.flop_counter import FlopCounterMode
    cfg = DiTConfig(dim=96, depth=1, heads=4, latent_channels=8, text_dim=16)
    T, H, W = 2, 4, 5
    N = T * H * W
    fc = FlopCounterMode(display=False)
    with torch.device("meta"):                 # on meta tensors SDPA is decomposed and counted too
        blk = DiTBlock(cfg).eval()
        cos, sin = rope_3d(T, H, W, cfg.head_dim)
        args = (torch.empty(1, N, 96), torch.empty(1, 6, 96), torch.empty(1, tool.TEXT_TOKENS, 96),
                torch.ones(1, tool.TEXT_TOKENS, dtype=torch.bool), cos, sin)
    with fc, torch.no_grad():
        blk(*args)
    m = tool.dit_forward_flops(cfg, N)
    io = 2 * N * 8 * 96 * 2 + 2 * tool.TEXT_TOKENS * 16 * 96
    assert abs((m["linear"] - io + m["attention"]) / fc.get_total_flops() - 1) < 0.01


def test_budget_quick_run(tmp_path):
    tool = load_tool()
    out = tmp_path / "m.md"
    assert tool.main(["--quick", "--out", str(out), "--threads", "2", "--json", str(tmp_path / "m.json")]) == 0
    text = out.read_text(encoding="utf-8")
    for needle in ("MEASURED on this CPU", "ESTIMATED: per-step FLOPs", "RTX 2060 6 GB", "932,333,696",
                   "Verdict", "untested on GPU", "28,520"):
        assert needle in text, needle
