#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data: training forward (loss), sampling
(get_action), tokenizer decode, and a step-count/latency sweep.

Usage:
    python scripts/smoke_test.py --ckpt checkpoints/ng.pt          # real weights
    python scripts/smoke_test.py --ckpt checkpoints/ng.pt --cpu    # no GPU
    python scripts/smoke_test.py                                   # tiny random model, no downloads*
      (*still fetches the SigLIP config from HF the first time)
"""

import argparse

import torch

from ngd.data.synthetic import make_synthetic_batch
from ngd.data.tokenizer import NitroGenTokenizer, TokenizerConfig
from ngd.eval.openloop import benchmark_sampling
from ngd.model.dit import DiTConfig, SelfAttentionTransformerConfig
from ngd.model.loading import load_checkpoint
from ngd.model.nitrogen import NitroGenConfig, NitroGenModel


def tiny_model_and_tokenizer():
    """Randomly initialized mini NitroGen (same code paths, ~fraction of the size)."""
    # Constraint (same as ng.pt, just smaller): vision_hidden_size == VL mixer width, and
    # the DiT cross-attends into VL tokens, so cross_attention_dim = vision_hidden_size.
    config = NitroGenConfig(
        diffusion_model_cfg=DiTConfig(
            num_attention_heads=4,
            attention_head_dim=32,
            num_layers=2,
            interleave_self_attention=True,
            positional_embeddings=None,
            cross_attention_dim=768,
            output_dim=128,  # == hidden_size: the action decoder reads the DiT output
        ),
        vl_self_attention_cfg=SelfAttentionTransformerConfig(
            num_attention_heads=8,
            attention_head_dim=96,
            num_layers=1,
            positional_embeddings=None,
        ),
        hidden_size=128,
        action_dim=25,
        action_horizon=18,
        num_inference_timesteps=4,
        add_pos_embed=True,
        vision_encoder_name="google/siglip2-base-patch16-256",
        vision_hidden_size=768,
    )
    model = NitroGenModel(config, pretrained_vision=False)
    tokenizer = NitroGenTokenizer(
        TokenizerConfig(max_sequence_length=256, action_horizon=18, training=True)
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=None, help="Path to ng.pt; omit for a tiny random model")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", help="Autocast to bfloat16 (CUDA)")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    if args.ckpt:
        loaded = load_checkpoint(args.ckpt, device=device, verbose=True)
        model, tokenizer = loaded.model, loaded.tokenizer
    else:
        model, tokenizer = tiny_model_and_tokenizer()
        model.eval().to(device)
        print(f"Tiny random model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    batch = make_synthetic_batch(tokenizer, batch_size=args.batch_size, training=True, device=device)
    print("\nBatch shapes:", {k: tuple(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)})

    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16 and device == "cuda")

    # 1) Training-style forward: flow-matching loss
    with torch.no_grad(), autocast:
        out = model(batch)
    print(f"\n[forward] flow-matching loss on random data: {out['loss'].item():.4f}")

    # 2) Sampling + decode
    with autocast:
        sampled = model.get_action(batch)
    decoded = tokenizer.decode(sampled)
    print(f"[get_action] action_tensor: {tuple(sampled['action_tensor'].shape)}")
    print(f"[decode] buttons {tuple(decoded['buttons'].shape)}, "
          f"j_left {tuple(decoded['j_left'].shape)}, j_right {tuple(decoded['j_right'].shape)}")
    print(f"[decode] mean buttons pressed/frame: {decoded['buttons'].sum(-1).mean().item():.2f}")

    # 3) Step-count sweep: latency + deviation from the 16-step reference
    print("\n[benchmark] Euler steps vs latency / L2-vs-16-step-reference")
    with autocast:
        rows = benchmark_sampling(model, batch, step_counts=(1, 2, 4, 8, 16))
    for row in rows:
        print(f"  steps={row['steps']:3d}  latency={row['latency_s']*1000:8.1f} ms  "
              f"L2 vs ref={row['l2_vs_reference']:.3f}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
