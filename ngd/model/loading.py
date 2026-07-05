"""Checkpoint loading for the official ``ng.pt`` (and future student checkpoints).

``ng.pt`` layout:
    {
        "model":       state dict (493.6M params),
        "step", "epoch",
        "ckpt_config": {"experiment_name", "model_cfg", "tokenizer_cfg", "modality_cfg"},
    }

Notes on the released checkpoint (2026-01 release):
    - ``game_mapping_cfg`` is null: no game-ID conditioning, no ``game_embedding`` weights.
    - ``action_horizon`` is 18 (not the 16 stated in the model card), ``action_dim`` 25
      (21 real dims: 17 buttons + 2x2 joysticks, zero-padded to 25).
    - Vision tower is ``google/siglip2-large-patch16-256``.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, Field

from ngd.data.tokenizer import NitroGenTokenizer, TokenizerConfig
from ngd.model.nitrogen import NitroGenConfig, NitroGenModel


class ModalityConfig(BaseModel):
    """How training samples were cut from videos (needed to reproduce the data pipeline)."""

    frame_per_sample: int = 1  # context frames per sample
    frame_spacing: Optional[int] = None  # frames between context frames; None -> action_per_chunk
    action_per_chunk: int = 8
    action_shift: int = 1  # offset between frame[i] and the first action of its chunk
    action_interleaving: bool = False
    token_set: str = "new"

    def model_post_init(self, __context):
        if self.frame_spacing is None:
            object.__setattr__(self, "frame_spacing", self.action_per_chunk)
        assert self.action_shift >= 1, "action_shift must be >= 1 for correct action indexing"


class CheckpointConfig(BaseModel):
    experiment_name: str = Field(...)
    model_cfg: NitroGenConfig
    tokenizer_cfg: TokenizerConfig
    modality_cfg: ModalityConfig


@dataclass
class LoadedCheckpoint:
    model: NitroGenModel
    tokenizer: NitroGenTokenizer
    image_processor: object  # HF image processor for the vision tower
    config: CheckpointConfig
    step: Optional[int] = None
    epoch: Optional[int] = None


def load_checkpoint(
    path: str | Path,
    device: str = "cuda",
    verbose: bool = True,
) -> LoadedCheckpoint:
    """Load ``ng.pt`` into the ngd implementation and return everything needed for inference.

    The vision tower is built from config only (no SigLIP weight download) since the
    checkpoint overwrites every parameter; loading is strict.
    """
    from transformers import AutoImageProcessor

    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = CheckpointConfig.model_validate(checkpoint["ckpt_config"])

    if verbose:
        print(f"Loaded checkpoint config ({path.name}):")
        print(json.dumps(config.model_dump(), indent=2, default=str))

    game_mapping = _resolve_game_mapping(config.tokenizer_cfg, checkpoint["model"])

    model = NitroGenModel(config.model_cfg, game_mapping=game_mapping, pretrained_vision=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)

    tokenizer = NitroGenTokenizer(config.tokenizer_cfg, game_mapping=game_mapping)
    tokenizer.eval()

    image_processor = AutoImageProcessor.from_pretrained(config.model_cfg.vision_encoder_name)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model ready on {device}: {n_params / 1e6:.1f}M parameters, "
              f"horizon={config.model_cfg.action_horizon}, "
              f"inference steps={config.model_cfg.num_inference_timesteps}")

    return LoadedCheckpoint(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        config=config,
        step=checkpoint.get("step"),
        epoch=checkpoint.get("epoch"),
    )


def _resolve_game_mapping(tokenizer_cfg: TokenizerConfig, state_dict: dict) -> Optional[dict]:
    """Reconstruct the game-name -> id mapping, or None if the checkpoint is unconditioned.

    The released ng.pt has no game conditioning. For checkpoints that do, the mapping is
    rebuilt from the parquet files referenced by ``game_mapping_cfg``; if those are not
    available locally, placeholder names sized from the embedding table are used so the
    weights still load.
    """
    has_embedding = "game_embedding.weight" in state_dict
    if tokenizer_cfg.game_mapping_cfg is None and not has_embedding:
        return None

    if tokenizer_cfg.game_mapping_cfg is not None:
        try:
            from ngd.data.tokenizer import build_game_mapping

            return build_game_mapping(tokenizer_cfg.game_mapping_cfg)
        except (FileNotFoundError, OSError) as err:
            print(f"Warning: game-mapping parquets unavailable ({err}); using placeholder names.")

    num_games = state_dict["game_embedding.weight"].shape[0]
    return {None: 0, **{f"game_{i:03d}": i for i in range(1, num_games)}}
