#!/usr/bin/env python3
"""Print a checkpoint's config and a parameter breakdown per top-level module.

Usage: python scripts/inspect_ckpt.py checkpoints/ng.pt
"""

import argparse
import json
from collections import defaultdict

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt", help="Path to checkpoint (.pt)")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print("Top-level keys:", list(checkpoint.keys()))
    print("step:", checkpoint.get("step"), " epoch:", checkpoint.get("epoch"))
    print("\n=== ckpt_config ===")
    print(json.dumps(checkpoint["ckpt_config"], indent=2, default=str))

    state_dict = checkpoint["model"]
    per_module = defaultdict(int)
    for key, tensor in state_dict.items():
        per_module[key.split(".")[0]] += tensor.numel()

    total = sum(per_module.values())
    print("\n=== parameters ===")
    for module, count in sorted(per_module.items(), key=lambda kv: -kv[1]):
        print(f"  {module:30s} {count / 1e6:8.1f}M  ({100 * count / total:5.1f}%)")
    print(f"  {'TOTAL':30s} {total / 1e6:8.1f}M")


if __name__ == "__main__":
    main()
