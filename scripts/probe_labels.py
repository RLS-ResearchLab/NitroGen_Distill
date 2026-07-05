#!/usr/bin/env python3
"""Probe the NitroGen HF dataset labels: download a sample of chunks and report the
statistics that matter for distillation (idle fraction, button press rates, right-stick
usage vs the virtual-button threshold, games covered).

Writes results/label_stats.json and .md.

Usage: python scripts/probe_labels.py [--videos 12] [--chunks-per-video 2] [--shard SHARD_0000]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from ngd.data.actions import DATASET_BUTTONS, load_chunk_actions, load_chunk_metadata

REPO = "nvidia/NitroGen"


def sample_chunk_dirs(api: HfApi, shard: str, n_videos: int, chunks_per_video: int) -> list[str]:
    """Return repo-relative chunk dirs like actions/SHARD_0000/<vid>/<vid>_chunk_0000."""
    videos = [e.path for e in api.list_repo_tree(REPO, f"actions/{shard}", repo_type="dataset")][:n_videos]
    chunk_dirs = []
    for video in videos:
        chunks = sorted(e.path for e in api.list_repo_tree(REPO, video, repo_type="dataset"))
        chunk_dirs.extend(chunks[:chunks_per_video])
    return chunk_dirs


def download_chunk(chunk_dir: str, dest_root: Path) -> Path | None:
    local = dest_root / chunk_dir
    for filename in ["metadata.json", "actions_processed.parquet", "actions_raw.parquet"]:
        try:
            hf_hub_download(REPO, f"{chunk_dir}/{filename}", repo_type="dataset",
                            local_dir=dest_root)
        except Exception:
            if filename == "metadata.json":
                return None  # unusable without metadata
    return local if any(local.glob("actions_*.parquet")) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default="SHARD_0000")
    parser.add_argument("--videos", type=int, default=12)
    parser.add_argument("--chunks-per-video", type=int, default=2)
    parser.add_argument("--dest", default="data/probe")
    parser.add_argument("--out", default="results")
    parser.add_argument("--stick-threshold", type=float, default=0.5)
    args = parser.parse_args()

    api = HfApi()
    chunk_dirs = sample_chunk_dirs(api, args.shard, args.videos, args.chunks_per_video)
    print(f"Sampling {len(chunk_dirs)} chunks from {args.shard}")

    dest_root = Path(args.dest)
    games, sources, controller_types = Counter(), Counter(), Counter()
    press = np.zeros(len(DATASET_BUTTONS))
    total_frames = 0
    idle_frames = 0
    stick_active = {"j_left": 0, "j_right": 0}
    right_virtual = np.zeros(4)  # up, down, left, right beyond threshold
    row_mismatches = []
    per_chunk = []

    for chunk_dir in chunk_dirs:
        local = download_chunk(chunk_dir, dest_root)
        if local is None:
            print(f"  skip (missing files): {chunk_dir}")
            continue
        meta = load_chunk_metadata(local)
        acts = load_chunk_actions(local)
        T = acts["buttons"].shape[0]

        if meta.get("chunk_size") != T:
            row_mismatches.append({"chunk": chunk_dir, "metadata": meta.get("chunk_size"), "parquet_rows": T})

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

        per_chunk.append({"chunk": chunk_dir, "game": meta.get("game"), "frames": T,
                          "fps": T / meta["original_video"]["duration"] if meta.get("original_video", {}).get("duration") else None,
                          "idle_frac": float(idle.mean())})

    assert total_frames > 0, "No chunks downloaded successfully"

    stats = {
        "shard": args.shard,
        "chunks": len(per_chunk),
        "total_frames": total_frames,
        "idle_frame_fraction": idle_frames / total_frames,
        "button_press_rate": {b: press[i] / total_frames for i, b in enumerate(DATASET_BUTTONS)},
        "stick_active_fraction": {k: v / total_frames for k, v in stick_active.items()},
        "right_stick_beyond_threshold": {
            "threshold": args.stick_threshold,
            "up": right_virtual[0] / total_frames, "down": right_virtual[1] / total_frames,
            "left": right_virtual[2] / total_frames, "right": right_virtual[3] / total_frames,
        },
        "games": dict(games), "sources": dict(sources), "controller_types": dict(controller_types),
        "metadata_vs_parquet_row_mismatches": row_mismatches,
        "per_chunk": per_chunk,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "label_stats.json").write_text(json.dumps(stats, indent=2))

    rates = sorted(stats["button_press_rate"].items(), key=lambda kv: -kv[1])
    lines = [
        f"# Label probe — {args.shard} ({stats['chunks']} chunks, {total_frames:,} frames)",
        "",
        f"- idle frames (no buttons, sticks in deadzone): **{stats['idle_frame_fraction']:.1%}**"
        " (paper mentions IDLE filtering — this is how much there is to filter)",
        f"- stick usage: left {stats['stick_active_fraction']['j_left']:.1%}, right {stats['stick_active_fraction']['j_right']:.1%}",
        f"- right stick beyond ±{args.stick_threshold} (virtual-button candidates): "
        + ", ".join(f"{k} {v:.2%}" for k, v in list(stats["right_stick_beyond_threshold"].items())[1:]),
        f"- games in sample: {', '.join(f'{g} ({n})' for g, n in games.most_common())}",
        f"- metadata/parquet row mismatches: {len(row_mismatches)}",
        "",
        "| button | press rate |",
        "|---|---|",
        *[f"| {b} | {r:.3%} |" for b, r in rates],
    ]
    (out_dir / "label_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))
    print(f"\nWrote {out_dir}/label_stats.json and .md")


if __name__ == "__main__":
    main()
