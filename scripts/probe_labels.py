#!/usr/bin/env python3
"""Probe extracted NitroGen dataset labels: statistics that matter for distillation
(idle fraction, button press rates, right-stick usage vs the virtual-button threshold,
games covered, metadata sanity).

Expects an extracted shard under --root (see ./env.sh dataset). Writes
results/label_stats.json and .md.

Usage: python scripts/probe_labels.py [--root data/nitrogen/actions] [--max-chunks 200]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ngd.data.actions import DATASET_BUTTONS, iter_chunk_dirs, load_chunk_actions, load_chunk_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/nitrogen/actions")
    parser.add_argument("--max-chunks", type=int, default=200)
    parser.add_argument("--out", default="results")
    parser.add_argument("--stick-threshold", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.root)
    assert root.exists(), f"{root} not found -- run ./env.sh dataset first"

    # Spread the sample across videos: take chunks round-robin by video directory.
    all_chunks = list(iter_chunk_dirs(root))
    by_video = Counter(c.parent for c in all_chunks)
    print(f"Found {len(all_chunks)} chunks across {len(by_video)} videos under {root}")
    chunks = sorted(all_chunks, key=lambda c: (int(c.name.rsplit("_", 1)[-1]) if c.name.rsplit("_", 1)[-1].isdigit() else 0, str(c)))
    chunks = chunks[: args.max_chunks]

    games, sources, controller_types = Counter(), Counter(), Counter()
    press = np.zeros(len(DATASET_BUTTONS))
    total_frames = 0
    idle_frames = 0
    stick_active = {"j_left": 0, "j_right": 0}
    right_virtual = np.zeros(4)  # up, down, left, right beyond threshold
    row_mismatches = []
    has_processed = 0
    per_chunk = []
    skipped = 0

    for chunk_dir in chunks:
        try:
            meta = load_chunk_metadata(chunk_dir)
            acts = load_chunk_actions(chunk_dir)
        except Exception as err:
            print(f"  skip {chunk_dir.name}: {err}")
            skipped += 1
            continue
        T = acts["buttons"].shape[0]

        if meta.get("chunk_size") != T:
            row_mismatches.append({"chunk": str(chunk_dir), "metadata": meta.get("chunk_size"), "parquet_rows": T})
        if (chunk_dir / "actions_processed.parquet").exists():
            has_processed += 1

        games[meta.get("game", "?")] += 1
        sources[meta.get("original_video", {}).get("source", "?")] += 1
        controller_types[meta.get("controller_type", "?")] += 1

        press += acts["buttons"].sum(axis=0)
        thr = args.stick_threshold
        jl, jr = acts["j_left"], acts["j_right"]
        stick_active["j_left"] += int((np.abs(jl) > 0.1).any(axis=1).sum())
        stick_active["j_right"] += int((np.abs(jr) > 0.1).any(axis=1).sum())
        right_virtual += np.array([
            (jr[:, 1] < -thr).sum(), (jr[:, 1] > thr).sum(), (jr[:, 0] < -thr).sum(), (jr[:, 0] > thr).sum(),
        ])
        idle = (acts["buttons"].sum(axis=1) == 0) & (np.abs(jl) <= 0.1).all(axis=1) & (np.abs(jr) <= 0.1).all(axis=1)
        idle_frames += int(idle.sum())
        total_frames += T

        duration = meta.get("original_video", {}).get("duration")
        per_chunk.append({"chunk": str(chunk_dir.relative_to(root)), "game": meta.get("game"), "frames": T,
                          "fps": T / duration if duration else None, "idle_frac": float(idle.mean())})

    assert total_frames > 0, "No chunks parsed successfully"

    stats = {
        "root": str(root),
        "chunks_sampled": len(per_chunk),
        "chunks_skipped": skipped,
        "chunks_with_processed_parquet": has_processed,
        "total_frames": total_frames,
        "idle_frame_fraction": idle_frames / total_frames,
        "button_press_rate": {b: press[i] / total_frames for i, b in enumerate(DATASET_BUTTONS)},
        "stick_active_fraction": {k: v / total_frames for k, v in stick_active.items()},
        "right_stick_beyond_threshold": {
            "threshold": args.stick_threshold,
            "up": right_virtual[0] / total_frames, "down": right_virtual[1] / total_frames,
            "left": right_virtual[2] / total_frames, "right": right_virtual[3] / total_frames,
        },
        "games": dict(games.most_common()), "sources": dict(sources), "controller_types": dict(controller_types),
        "metadata_vs_parquet_row_mismatches": row_mismatches[:20],
        "per_chunk": per_chunk,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "label_stats.json").write_text(json.dumps(stats, indent=2))

    rates = sorted(stats["button_press_rate"].items(), key=lambda kv: -kv[1])
    lines = [
        f"# Label probe — {stats['chunks_sampled']} chunks, {total_frames:,} frames",
        "",
        f"- idle frames (no buttons, sticks in deadzone): **{stats['idle_frame_fraction']:.1%}**"
        " (paper mentions IDLE filtering — this is how much there is to filter)",
        f"- chunks with actions_processed.parquet: {has_processed}/{stats['chunks_sampled']}",
        f"- stick usage: left {stats['stick_active_fraction']['j_left']:.1%}, right {stats['stick_active_fraction']['j_right']:.1%}",
        f"- right stick beyond ±{args.stick_threshold} (virtual-button candidates): "
        + ", ".join(f"{k} {v:.2%}" for k, v in list(stats["right_stick_beyond_threshold"].items())[1:]),
        f"- games in sample ({len(games)}): {', '.join(f'{g} ({n})' for g, n in games.most_common(15))}",
        f"- metadata/parquet row mismatches: {len(row_mismatches)}",
        "",
        "| button | press rate |",
        "|---|---|",
        *[f"| {b} | {r:.3%} |" for b, r in rates],
    ]
    (out_dir / "label_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"\nWrote {out_dir}/label_stats.json and .md")


if __name__ == "__main__":
    main()
