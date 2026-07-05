"""Tokenizer layout correctness: pack/unpack round-trips, masks, padding."""

import numpy as np
import torch

from ngd.constants import ACT_TOKEN, IMG_TOKEN, PAD_TOKEN
from ngd.data.tokenizer import NitroGenTokenizer, TokenizerConfig

CFG = TokenizerConfig(max_sequence_length=256, action_horizon=18, max_action_dim=25, training=True)


def test_pack_unpack_roundtrip():
    tok = NitroGenTokenizer(CFG)
    rng = np.random.default_rng(0)
    buttons = rng.integers(0, 2, (1, 18, 21)).astype(np.float32)
    j_left = rng.uniform(-1, 1, (1, 18, 2)).astype(np.float32)
    j_right = rng.uniform(-1, 1, (1, 18, 2)).astype(np.float32)

    packed = tok.pack_actions(buttons, j_left, j_right)
    assert packed.shape == (18, 25)
    # layout: [buttons(21), j_left(2), j_right(2)], sticks mapped [-1,1] -> [0,1]
    assert np.allclose(packed[:, :21], buttons[0])
    assert packed[:, 21:].min() >= 0.0 and packed[:, 21:].max() <= 1.0

    jl, jr, b = tok.unpack_actions(torch.from_numpy(packed).unsqueeze(0))
    assert torch.allclose(jl[0], torch.from_numpy(j_left[0]), atol=1e-6)
    assert torch.allclose(jr[0], torch.from_numpy(j_right[0]), atol=1e-6)
    assert torch.equal(b[0], torch.from_numpy(buttons[0]))  # 0/1 inputs survive thresholding


def test_encode_training_shapes_and_masks():
    tok = NitroGenTokenizer(CFG)
    rng = np.random.default_rng(1)
    out = tok.encode(
        {
            "frames": torch.randn(1, 3, 256, 256),
            "dropped_frames": torch.zeros(1, dtype=torch.bool),
            "game": None,
            "buttons": rng.integers(0, 2, (1, 18, 21)).astype(np.float32),
            "j_left": rng.uniform(-1, 1, (1, 18, 2)).astype(np.float32),
            "j_right": rng.uniform(-1, 1, (1, 18, 2)).astype(np.float32),
        }
    )
    assert out["actions"].shape == (18, 25)
    assert out["actions_mask"].shape == (18, 25)
    assert out["actions_mask"].all()  # 25 real dims: no padding for ng.pt's action space
    assert out["vl_token_ids"].shape == (256,)
    assert (out["vl_token_ids"] == IMG_TOKEN).sum() == 256  # 1 frame x 256 visual tokens
    assert (out["sa_token_ids"] == ACT_TOKEN).sum() == 18
    assert out["vl_attn_mask"].all()


def test_vl_left_padding_with_fewer_tokens():
    cfg = CFG.model_copy(update={"max_sequence_length": 300})
    tok = NitroGenTokenizer(cfg)
    tok.eval()
    out = tok.encode({"frames": torch.randn(1, 3, 256, 256), "dropped_frames": torch.zeros(1, dtype=torch.bool), "game": None})
    assert out["vl_token_ids"].shape == (300,)
    assert (out["vl_token_ids"][:44] == PAD_TOKEN).all()  # left-padded
    assert (out["vl_token_ids"][44:] == IMG_TOKEN).all()
    assert not out["vl_attn_mask"][:44].any() and out["vl_attn_mask"][44:].all()
