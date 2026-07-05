"""Torch dataset producing model-ready samples from (video file, action chunk) pairs.

A sample follows the checkpoint's modality config (ng.pt: frame_per_sample=1,
action_per_chunk=18, action_shift=3): one context frame at chunk index ``t`` paired with
the action rows ``[t + action_shift, t + action_shift + horizon)``.

Decoding frames straight from video files is convenient for evaluation but slow for
training (one seek per sample) -- for training runs, pre-extract frames to disk and swap
the frame source.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ngd.data.actions import dataset_to_model_actions, load_chunk_actions, load_chunk_metadata
from ngd.data.tokenizer import NitroGenTokenizer
from ngd.data.video import extract_chunk_frames


@dataclass
class ChunkSource:
    chunk_dir: Path  # directory with actions_*.parquet + metadata.json
    video_path: Path  # local video file covering this chunk


class ChunkDataset(Dataset):
    """Iterates (frame, action-chunk) samples over a list of labeled chunks."""

    def __init__(
        self,
        sources: list[ChunkSource],
        tokenizer: NitroGenTokenizer,
        image_processor,  # HF processor from the loaded checkpoint
        action_horizon: int = 18,
        action_shift: int = 3,
        frame_spacing: int = 18,  # stride between consecutive samples' context frames
        prefer_processed: bool = True,
    ):
        self.sources = sources
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.action_horizon = action_horizon
        self.action_shift = action_shift
        self.frame_spacing = frame_spacing
        self.prefer_processed = prefer_processed

        # Flat index: (source_idx, chunk-frame index of the context frame).
        self.index: list[tuple[int, int]] = []
        for src_idx, src in enumerate(sources):
            meta = load_chunk_metadata(src.chunk_dir)
            last_start = meta["chunk_size"] - action_shift - action_horizon
            for t in range(0, max(last_start + 1, 0), frame_spacing):
                self.index.append((src_idx, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        src_idx, t = self.index[i]
        src = self.sources[src_idx]
        meta = load_chunk_metadata(src.chunk_dir)
        actions = load_chunk_actions(src.chunk_dir, prefer_processed=self.prefer_processed)

        frames = dict(extract_chunk_frames(src.video_path, meta, frame_indices=[t], size=None))
        frame = frames[t]  # RGB uint8
        pixel_values = self.image_processor([frame], return_tensors="pt")["pixel_values"]  # (1, C, H, W)

        sl = slice(t + self.action_shift, t + self.action_shift + self.action_horizon)
        # Dataset labels 17 physical buttons; the model's action space has 21 (with
        # virtual right-stick buttons) -- see ngd.data.actions for the caveat.
        model_buttons = dataset_to_model_actions(actions["buttons"][sl], actions["j_right"][sl])
        sample = {
            "frames": pixel_values,
            "dropped_frames": torch.zeros(pixel_values.shape[0], dtype=torch.bool),
            "game": meta.get("game"),
            # pack_actions expects a leading chunk dim of 1
            "buttons": model_buttons[None],
            "j_left": actions["j_left"][None, sl],
            "j_right": actions["j_right"][None, sl],
        }
        return self.tokenizer.encode(sample)


def collate_samples(samples: list[dict], device: str | torch.device = "cpu") -> dict:
    """Stack tokenizer.encode outputs into the batched dict NitroGenModel consumes."""

    def stack(key, dtype=None):
        vals = []
        for s in samples:
            v = s[key]
            v = torch.as_tensor(np.asarray(v)) if not isinstance(v, torch.Tensor) else v
            vals.append(v)
        out = torch.stack(vals).to(device)
        return out.to(dtype) if dtype is not None else out

    batch = {
        "images": stack("images", torch.float32),
        "dropped_images": stack("dropped_images"),
        "vl_token_ids": stack("vl_token_ids", torch.long),
        "sa_token_ids": stack("sa_token_ids", torch.long),
        "vl_attn_mask": stack("vl_attn_mask"),
        "embodiment_id": stack("embodiment_id"),
        "game_ids": stack("game_ids"),
    }
    if "actions" in samples[0]:
        batch["actions"] = stack("actions", torch.float32)
        batch["actions_mask"] = stack("actions_mask")
        batch["has_real_action"] = stack("has_real_action")
    return batch
