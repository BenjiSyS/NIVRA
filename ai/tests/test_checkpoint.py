import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nivra.config import build_gen
from nivra.data.synthetic import SynthConfig
from nivra.models.nivra_small import ARCH_VERSION
from nivra.text.tokenizer import default_tokenizer
from nivra.training import checkpoint as ck
from nivra.training.tasks import GenTask
from nivra.training.trainer import Trainer, TrainerConfig


def make_trainer(cfg, run_dir, total=6, init_seed=0, **over):
    torch.manual_seed(init_seed)
    model = build_gen(cfg)
    g = torch.Generator().manual_seed(123)
    lat = torch.randn(24, 4, 4, 8, 8, generator=g)
    tok = default_tokenizer()
    ids = torch.tensor([[tok.encode("a red circle moves up") for _ in range(4)] for _ in range(24)])
    task = GenTask(model, lat, ids, SynthConfig(frames=8, size=32), "vae-sha")
    kw = dict(total_steps=total, batch_size=4, lr=1e-3, warmup=2, ema_decay=0.9, save_every_steps=0,
              save_every_minutes=0, log_every=100, seed=5)
    kw.update(over)
    compat = {"kind": "gen", "arch_version": ARCH_VERSION, "tokenizer_version": "x",
              "vocab_hash": default_tokenizer().vocab_hash}
    return Trainer(task, TrainerConfig(**kw), run_dir, compat, cfg, stop_file=Path(run_dir) / "STOP")


def _assert_same(a, c, total):
    for (n, pa), (_, pc) in zip(a.model.named_parameters(), c.model.named_parameters()):
        assert torch.equal(pa, pc), n
    for k in a.ema.shadow:
        assert torch.equal(a.ema.shadow[k], c.ema.shadow[k]), k
    sa, sc = a.opt.state_dict()["state"], c.opt.state_dict()["state"]
    assert len(sa) == len(sc)
    for i in sa:
        assert float(sa[i]["step"]) == float(sc[i]["step"]) == total
        assert torch.equal(sa[i]["exp_avg"], sc[i]["exp_avg"])
        assert torch.equal(sa[i]["exp_avg_sq"], sc[i]["exp_avg_sq"])
    assert np.array_equal(a.sampler.next(), c.sampler.next())          # next-batch identity
    assert torch.equal(a.gen.get_state(), c.gen.get_state())
    assert a.sched.state_dict() == c.sched.state_dict()


def test_resume_is_exactly_equivalent_to_uninterrupted(tiny_cfg, tmp_path):
    a = make_trainer(tiny_cfg, tmp_path / "a")
    ra = a.run(install_signals=False)
    assert ra["status"] == "done" and a.step == 6
    b = make_trainer(tiny_cfg, tmp_path / "b")
    rb = b.run(max_steps=3, install_signals=False)
    assert rb["status"] == "paused" and b.step == 3
    b.save()
    c = make_trainer(tiny_cfg, tmp_path / "b", init_seed=999)       # different init: must be fully overwritten
    assert c.resume() and c.step == 3
    assert c.sampler.state_dict() == b.sampler.state_dict()
    c.run(install_signals=False)
    assert c.step == 6
    _assert_same(a, c, 6)


def test_resume_across_epoch_boundary(tiny_cfg, tmp_path):
    # 24 clips / batch 4 = 6 batches per epoch -> 9 steps cross a boundary
    a = make_trainer(tiny_cfg, tmp_path / "a", total=9)
    a.run(install_signals=False)
    b = make_trainer(tiny_cfg, tmp_path / "b", total=9)
    b.run(max_steps=7, install_signals=False)
    b.save()
    c = make_trainer(tiny_cfg, tmp_path / "b", total=9, init_seed=3)
    assert c.resume() and c.sampler.epoch == 1
    c.run(install_signals=False)
    _assert_same(a, c, 9)


def test_stop_file_saves_and_stops(tiny_cfg, tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "STOP").write_text("x")
    t = make_trainer(tiny_cfg, tmp_path / "s")
    r = t.run(install_signals=False)
    assert r["status"] == "stopped"
    assert ck.newest_valid(tmp_path / "s") is not None


def _saved(tiny_cfg, tmp_path):
    t = make_trainer(tiny_cfg, tmp_path / "x")
    t.run(max_steps=2, install_signals=False)
    return t, t.save()


def test_checkpoint_files_and_no_temp_leftovers(tiny_cfg, tmp_path):
    t, p = _saved(tiny_cfg, tmp_path)
    names = {f.name for f in p.iterdir()}
    assert {"manifest.json", "model.safetensors", "ema.safetensors", "optimizer.safetensors", "rng.safetensors",
            "state.json"} <= names
    assert not [d for d in (tmp_path / "x").iterdir() if d.name.startswith(".tmp")]
    m = ck.verify_checkpoint(p)
    assert m["compat"]["dataset_fingerprint"] == t.task.fingerprint and m["step"] == 2
    assert all(len(v["sha256"]) == 64 for v in m["files"].values())


@pytest.mark.parametrize("victim", ["model.safetensors", "optimizer.safetensors", "state.json"])
def test_flipped_byte_is_refused(tiny_cfg, tmp_path, victim):
    t, p = _saved(tiny_cfg, tmp_path)
    f = p / victim
    data = bytearray(f.read_bytes())
    data[len(data) // 2] ^= 0x01
    f.write_bytes(bytes(data))
    with pytest.raises(ck.CheckpointError):
        ck.load_checkpoint(p)
    fresh = make_trainer(tiny_cfg, tmp_path / "x")
    with pytest.raises(ck.CheckpointError):
        fresh.resume(p)


def test_tampered_manifest_is_refused(tiny_cfg, tmp_path):
    t, p = _saved(tiny_cfg, tmp_path)
    m = json.loads((p / "manifest.json").read_text())
    m["step"] = 999
    (p / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ck.CheckpointError):
        ck.verify_checkpoint(p)


def test_incompatible_architecture_or_tokenizer_refused(tiny_cfg, tmp_path):
    t, p = _saved(tiny_cfg, tmp_path)
    with pytest.raises(ck.IncompatibleCheckpoint):
        ck.load_checkpoint(p, expect_compat={"arch_version": "nivra-small/2"})
    with pytest.raises(ck.IncompatibleCheckpoint):
        ck.load_checkpoint(p, expect_compat={"vocab_hash": "deadbeef"})
    ck.load_checkpoint(p, expect_compat={"arch_version": ARCH_VERSION})


def test_retention_keeps_newest_valid_and_protected(tmp_path):
    run = tmp_path / "r"
    for s in (1, 2, 3, 4, 5):
        ck.save_checkpoint(run, s, {"model": {"w": torch.ones(2) * s}}, {}, {"kind": "t"}, {})
    ck.protect(run / "ckpt_00000002")
    f = run / "ckpt_00000005" / "model.safetensors"       # corrupt the newest: newest *valid* is step 4
    b = bytearray(f.read_bytes())
    b[-1] ^= 1
    f.write_bytes(bytes(b))
    deleted = ck.apply_retention(run, keep_last=1)
    left = {p.name for _, p in ck.list_checkpoints(run)}
    assert {"ckpt_00000004", "ckpt_00000002", "ckpt_00000005"} <= left
    assert {d.name for d in deleted} == {"ckpt_00000001", "ckpt_00000003"}
    assert ck.newest_valid(run).name == "ckpt_00000004"


def test_periodic_save_by_steps_and_protect_steps(tiny_cfg, tmp_path):
    t = make_trainer(tiny_cfg, tmp_path / "p", protect_steps=(2,), save_every_steps=3, keep_last=1)
    t.run(install_signals=False)
    names = {p.name for _, p in ck.list_checkpoints(tmp_path / "p")}
    assert "ckpt_00000002" in names and "ckpt_00000006" in names and "ckpt_00000003" not in names
