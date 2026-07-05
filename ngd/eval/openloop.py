"""Open-loop evaluation: compare predicted action chunks against ground-truth labels,
and benchmark sampling cost/quality across Euler step counts.

Open-loop (per-frame prediction vs. logged action) is the cheap proxy metric; it does not
capture compounding closed-loop drift, but it is the right harness for comparing teacher
vs. distilled students on identical inputs.
"""

import time

import torch


@torch.no_grad()
def action_metrics(
    pred: dict[str, torch.Tensor],  # decoded: buttons (B,T,17) 0/1, j_left/j_right (B,T,2)
    target: dict[str, torch.Tensor],  # same layout, ground truth
) -> dict[str, float]:
    """Button F1/precision/recall + joystick MAE between two decoded action dicts."""
    p, t = pred["buttons"].bool(), target["buttons"].bool()
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "button_precision": precision,
        "button_recall": recall,
        "button_f1": f1,
        "button_press_rate_pred": p.float().mean().item(),
        "button_press_rate_target": t.float().mean().item(),
        "j_left_mae": (pred["j_left"] - target["j_left"]).abs().mean().item(),
        "j_right_mae": (pred["j_right"] - target["j_right"]).abs().mean().item(),
    }


@torch.no_grad()
def benchmark_sampling(
    model,
    batch: dict,
    step_counts: tuple[int, ...] = (1, 2, 4, 8, 16),
    reference_steps: int = 16,
    seed: int = 0,
    warmup: int = 1,
) -> list[dict]:
    """Latency and deviation-from-reference for different Euler step counts.

    Deviation is the L2 distance between actions sampled with N steps and with
    ``reference_steps``, from the same initial noise — a direct read on how much
    step-distillation headroom exists before any training.
    """
    device = model.device
    results = []

    def sample(n):
        torch.manual_seed(seed)  # same initial noise across step counts
        return model.get_action(batch, num_inference_timesteps=n)["action_tensor"]

    for _ in range(warmup):
        sample(1)

    reference = sample(reference_steps)

    for n in step_counts:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        actions = sample(n)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - start

        results.append(
            {
                "steps": n,
                "latency_s": latency,
                "l2_vs_reference": (actions - reference).norm().item(),
            }
        )
    return results
