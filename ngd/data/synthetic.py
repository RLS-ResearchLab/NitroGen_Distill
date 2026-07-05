"""Synthetic batches: exercise the full model pipeline with zero data downloads.

Frames are random noise and actions are random button presses / stick positions, so the
outputs are meaningless -- but shapes, masks, the tokenizer round-trip, the training
loss, and the sampling loop are all exercised exactly as with real data. Useful for
smoke tests, latency benchmarks, and unit-testing distillation plumbing.
"""

import numpy as np
import torch

from ngd.data.dataset import collate_samples
from ngd.data.tokenizer import NitroGenTokenizer


def make_synthetic_batch(
    tokenizer: NitroGenTokenizer,
    batch_size: int = 2,
    n_frames: int = 1,
    image_size: int = 256,
    n_buttons: int = 21,  # model action space: 21 buttons + 2x2 sticks = 25 dims
    training: bool = True,
    device: str | torch.device = "cpu",
    seed: int | None = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    horizon = tokenizer.action_horizon
    was_training = tokenizer.training
    tokenizer.training = training

    samples = []
    for _ in range(batch_size):
        sample = {
            "frames": torch.randn(n_frames, 3, image_size, image_size),
            "dropped_frames": torch.zeros(n_frames, dtype=torch.bool),
            "game": None,
        }
        if training:
            sample["buttons"] = rng.integers(0, 2, (1, horizon, n_buttons)).astype(np.float32)
            sample["j_left"] = rng.uniform(-1, 1, (1, horizon, 2)).astype(np.float32)
            sample["j_right"] = rng.uniform(-1, 1, (1, horizon, 2)).astype(np.float32)
        samples.append(tokenizer.encode(sample))

    tokenizer.training = was_training
    return collate_samples(samples, device=device)
