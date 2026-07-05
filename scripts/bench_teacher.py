#!/usr/bin/env python3
"""Teacher inference benchmark: latency/throughput vs Euler steps, batch size, dtype,
plus deviation-from-16-step-reference per step count (step-distillation headroom).

Writes results/bench_teacher.json and prints a markdown summary.

Usage: python scripts/bench_teacher.py [--ckpt checkpoints/ng.pt] [--out results]
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from ngd.data.synthetic import make_synthetic_batch
from ngd.model.loading import load_checkpoint


@torch.no_grad()
def bench(model, batch, steps, dtype, iters=5, seed=0):
    device = model.device
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dtype == "bf16" and device.type == "cuda")

    def sample():
        torch.manual_seed(seed)
        with autocast:
            return model.get_action(batch, num_inference_timesteps=steps)["action_tensor"]

    sample()  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        actions = sample()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    t = torch.tensor(times)
    return actions, {"mean_ms": t.mean().item() * 1000, "std_ms": t.std().item() * 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="checkpoints/ng.pt")
    parser.add_argument("--out", default="results")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_checkpoint(args.ckpt, device=device, verbose=False)
    model, tokenizer = loaded.model, loaded.tokenizer
    horizon = loaded.config.model_cfg.action_horizon

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()
    print(f"Benchmarking on {gpu} ({device})\n")

    records = []
    for dtype in (["bf16", "fp32"] if device == "cuda" else ["fp32"]):
        for bs in args.batch_sizes:
            batch = make_synthetic_batch(tokenizer, batch_size=bs, training=True, device=device, seed=0)
            reference, _ = bench(model, batch, 16, dtype, iters=1)
            for steps in args.steps:
                actions, timing = bench(model, batch, steps, dtype, iters=args.iters)
                per_dim_dev = (actions - reference).abs().mean().item()
                records.append(
                    {
                        "dtype": dtype,
                        "batch_size": bs,
                        "steps": steps,
                        **timing,
                        "actions_per_s": bs * horizon / (timing["mean_ms"] / 1000),
                        "mean_abs_dev_vs_16step": per_dim_dev,
                    }
                )
                r = records[-1]
                print(f"  {dtype} bs={bs:3d} steps={steps:3d}  {r['mean_ms']:8.1f}±{r['std_ms']:.1f} ms"
                      f"  {r['actions_per_s']:9.0f} actions/s  dev={per_dim_dev:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    payload = {"gpu": gpu, "torch": torch.__version__, "horizon": horizon, "records": records}
    (out_dir / "bench_teacher.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"# Teacher benchmark — {gpu}",
        "",
        f"torch {torch.__version__}, horizon {horizon}, synthetic frames, {args.iters} iters/point.",
        "`dev` = mean |Δ| per action dim vs the 16-step reference from the same noise.",
        "",
        "| dtype | batch | steps | latency (ms) | actions/s | dev vs 16-step |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['dtype']} | {r['batch_size']} | {r['steps']} | {r['mean_ms']:.1f} ± {r['std_ms']:.1f} "
            f"| {r['actions_per_s']:.0f} | {r['mean_abs_dev_vs_16step']:.4f} |"
        )
    (out_dir / "bench_teacher.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_dir}/bench_teacher.json and .md")


if __name__ == "__main__":
    main()
