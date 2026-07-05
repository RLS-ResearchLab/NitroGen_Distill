"""Model-level correctness on a tiny random NitroGen (same code paths as ng.pt)."""

import torch

from ngd.data.synthetic import make_synthetic_batch


def test_forward_loss_finite_and_masked(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    batch = make_synthetic_batch(tokenizer, batch_size=2, training=True, seed=0)
    with torch.no_grad():
        out = model(batch)
    assert out["loss"].isfinite()
    assert out["loss"].ndim == 0


def test_get_action_shapes_and_determinism(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    batch = make_synthetic_batch(tokenizer, batch_size=2, training=True, seed=0)

    torch.manual_seed(123)
    a1 = model.get_action(batch)["action_tensor"]
    torch.manual_seed(123)
    a2 = model.get_action(batch)["action_tensor"]

    assert a1.shape == (2, 18, 25)
    assert torch.equal(a1, a2)  # eval mode + same seed => identical sampling


def test_step_count_override(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    batch = make_synthetic_batch(tokenizer, batch_size=1, training=True, seed=0)
    torch.manual_seed(0)
    a1 = model.get_action(batch, num_inference_timesteps=1)["action_tensor"]
    torch.manual_seed(0)
    a16 = model.get_action(batch, num_inference_timesteps=16)["action_tensor"]
    assert a1.shape == a16.shape
    assert not torch.equal(a1, a16)  # different integrators, same noise


def test_decode_action_space(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    batch = make_synthetic_batch(tokenizer, batch_size=1, training=True, seed=0)
    decoded = tokenizer.decode(model.get_action(batch))
    assert decoded["buttons"].shape == (1, 18, 21)
    assert decoded["j_left"].shape == (1, 18, 2)
    assert decoded["j_right"].shape == (1, 18, 2)
    assert set(decoded["buttons"].unique().tolist()) <= {0.0, 1.0}
    assert decoded["j_left"].abs().max() <= 1.0 and decoded["j_right"].abs().max() <= 1.0
