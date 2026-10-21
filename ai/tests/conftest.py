import os
import sys
from pathlib import Path

os.environ.setdefault("NIVRA_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

torch.set_num_threads(int(os.environ["NIVRA_THREADS"]))


@pytest.fixture(scope="session")
def tiny_cfg():
    from nivra.config import load_config
    return load_config("nivra_tiny")


@pytest.fixture(scope="session")
def tiny_env(tmp_path_factory, tiny_cfg):
    """Tiny VAE (4 training steps) + tiny random-init generator version, shared by tests."""
    from nivra.training.pipeline import train_vae
    from nivra.inference.bundle import init_random_bundle
    root = tmp_path_factory.mktemp("tiny")
    res = train_vae(tiny_cfg, root / "vae", 4, resume=False, log=lambda *_: None, save_every_steps=2)
    assert res["status"] == "done"
    vae = Path(res["vae_path"])
    bundle = root / "bundle0"
    m = init_random_bundle(tiny_cfg, vae, bundle, seed=0)
    return {"root": root, "vae": vae, "bundle": bundle, "manifest": m}
