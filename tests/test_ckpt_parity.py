"""Numerical parity: ngd reimplementation vs the official implementation, same ng.pt.

Loads the checkpoint into both models and compares (in fp32, eval mode, same RNG seed):
    1. vision encoding of identical frames,
    2. the flow-matching training loss on an identical batch,
    3. full 16-step sampled action chunks.

Requires checkpoints/ng.pt and third_party/NitroGen (both handled by ./env.sh).
The official model's __init__ downloads SigLIP2 weights once; they are immediately
overwritten by the checkpoint load.
"""

import pytest
import torch

from ngd.data.synthetic import make_synthetic_batch
from tests.conftest import CKPT_PATH, needs_ckpt, needs_official, official_repo_on_path

pytestmark = [needs_ckpt, needs_official]

ATOL = 1e-4  # fp32, identical op order; report actual max diff on failure


@pytest.fixture(scope="module")
def both_models(device):
    official_repo_on_path()
    from nitrogen.flow_matching_transformer.nitrogen import NitroGen as OfficialNitroGen
    from nitrogen.flow_matching_transformer.nitrogen import NitroGen_Config as OfficialConfig

    from ngd.model.loading import load_checkpoint

    loaded = load_checkpoint(CKPT_PATH, device=device, verbose=False)
    ours = loaded.model

    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    official_cfg = OfficialConfig.model_validate(checkpoint["ckpt_config"]["model_cfg"])
    official = OfficialNitroGen(config=official_cfg, game_mapping=None)
    official.load_state_dict(checkpoint["model"], strict=True)
    official.eval().to(device)

    batch = make_synthetic_batch(loaded.tokenizer, batch_size=2, training=True, seed=7, device=device)
    batch["game_id"] = batch["game_ids"]  # official forward reads "game_id", get_action "game_ids"
    return ours, official, batch


def _report(name, a, b):
    diff = (a - b).abs().max().item()
    print(f"{name}: max abs diff = {diff:.3e}")
    return diff


def test_state_dicts_identical(both_models):
    ours, official, _ = both_models
    ours_sd, official_sd = ours.state_dict(), official.state_dict()
    assert set(ours_sd) == set(official_sd)
    for key in ours_sd:
        assert torch.equal(ours_sd[key], official_sd[key]), key


def test_vision_encoding_parity(both_models):
    ours, official, batch = both_models
    with torch.no_grad():
        a = ours.encode_images(batch["images"])
        b = official.encode_images(batch["images"])
    assert _report("encode_images", a, b) <= ATOL


def test_training_loss_parity(both_models):
    ours, official, batch = both_models
    with torch.no_grad():
        torch.manual_seed(42)
        loss_ours = ours(batch)["loss"]
        torch.manual_seed(42)
        loss_official = official(batch)["loss"]
    assert _report("forward loss", loss_ours, loss_official) <= ATOL


def test_sampling_parity(both_models):
    ours, official, batch = both_models
    with torch.no_grad():
        torch.manual_seed(42)
        a = ours.get_action(batch)["action_tensor"]
        torch.manual_seed(42)
        b = official.get_action(batch)["action_tensor"]
    assert _report("get_action (16 steps)", a, b) <= ATOL
