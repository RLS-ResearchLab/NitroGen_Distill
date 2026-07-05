"""Tokenizer: packs frames + actions into the tensor dict consumed by NitroGenModel.

"Tokenization" here is layout work, not learned encoding:
    - pack buttons + joysticks into one float action vector, zero-padded to max_action_dim;
    - build token-type id sequences (which slots are image tokens, game-ID token, action
      tokens) plus attention/padding masks;
    - decode: split the model's sampled action tensor back into buttons / joysticks.

Action layout ("new", used by ng.pt): [buttons(21), j_left(2), j_right(2)] = 25 dims,
where the 21 buttons are MODEL_BUTTONS from ngd.data.actions (17 physical + 4 virtual
right-stick directions). Joysticks are normalized [-1, 1] -> [0, 1] inside the packed
vector. max_action_dim padding only applies if fewer dims are supplied.
"""

from typing import Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field

from ngd.constants import ACT_TOKEN, GAME_ID_TOKEN, IMG_TOKEN, PAD_TOKEN

UNCONDITIONAL_ID = None  # game-mapping key reserved for the unconditional embedding (id 0)


class GameMappingConfig(BaseModel):
    src_files: list[str] = Field(default_factory=list, description="Parquet files with a 'game_label' column.")


def build_game_mapping(cfg: GameMappingConfig) -> dict:
    """Build {game_name: id} from parquet game labels; id 0 is the unconditional slot."""
    import polars as pl

    game_set = set()
    for path in cfg.src_files:
        for game in pl.read_parquet(path)["game_label"].unique():
            if game != UNCONDITIONAL_ID:
                game_set.add(game)
    games = [UNCONDITIONAL_ID] + sorted(game_set)
    return {game: idx for idx, game in enumerate(games)}


class TokenizerConfig(BaseModel):
    """Field names mirror the official NitrogenTokenizerConfig (ckpt_config compatibility)."""

    tokenizer_id: Literal["nitrogen"] = Field(default="nitrogen", frozen=True)
    training: bool = Field(default=True)
    num_visual_tokens_per_frame: int = Field(default=256)
    max_action_dim: int = Field(default=25)
    max_sequence_length: int = Field(default=300)
    action_horizon: int = Field(default=16)
    game_mapping_cfg: Optional[GameMappingConfig] = Field(default=None)
    use_action_mask: bool = Field(default=True)
    old_layout: bool = Field(default=False, description="True: [j_left, j_right, buttons]; False: [buttons, j_left, j_right].")


class NitroGenTokenizer:
    def __init__(self, config: TokenizerConfig, game_mapping: Optional[dict] = None):
        self.config = config
        self.training = config.training
        self.num_visual_tokens_per_frame = config.num_visual_tokens_per_frame
        self.max_action_dim = config.max_action_dim
        self.max_sequence_length = config.max_sequence_length
        self.action_horizon = config.action_horizon
        self.old_layout = config.old_layout

        # The mapping is injected (built once by the loader / training setup), not read
        # from parquet here, so the tokenizer has no filesystem dependency.
        self.game_mapping = game_mapping

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    # ------------------------------------------------------------------ packing

    def pack_actions(self, buttons: np.ndarray, j_left: np.ndarray, j_right: np.ndarray) -> np.ndarray:
        """(chunks, T, 17), (chunks, T, 2), (chunks, T, 2) -> (T, 21) float32.

        Joysticks are mapped [-1,1] -> [0,1] so every action dim lives in [0,1].
        """
        assert buttons.shape[:2] == j_left.shape[:2] == j_right.shape[:2], (
            f"{buttons.shape=}, {j_left.shape=}, {j_right.shape=}"
        )
        j_left = (j_left + 1) / 2.0
        j_right = (j_right + 1) / 2.0
        action = np.concatenate([buttons, j_left, j_right], axis=-1, dtype=np.float32)
        return action.squeeze(0)  # single chunk

    def unpack_actions(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, T, >=21) sampled actions -> (j_left, j_right, buttons); thresholds buttons at 0.5."""
        if self.old_layout:
            j_left, j_right, buttons = actions[:, :, :2], actions[:, :, 2:4], actions[:, :, 4:]
        else:
            buttons, j_left, j_right = actions[:, :, :-4], actions[:, :, -4:-2], actions[:, :, -2:]

        j_left = torch.clamp(j_left * 2.0 - 1.0, -1, 1)
        j_right = torch.clamp(j_right * 2.0 - 1.0, -1, 1)
        buttons = (buttons > 0.5).float()
        return j_left, j_right, buttons

    # ------------------------------------------------------------------ encode/decode

    def encode(self, data: dict) -> dict:
        """Prepare one (unbatched) sample.

        Input keys:
            frames          (F, C, H, W) preprocessed frames (image-processor output)
            dropped_frames  (F,) bool, True where the frame slot is empty padding
            game            str | None, game name for conditioning
            buttons/j_left/j_right   (training only) raw action chunk, shape (1, T, dim)

        Output: dict ready for NitroGenModel (add batch dim + device before use).
        """
        out = {**data}
        out["images"] = data["frames"]
        out["dropped_images"] = data["dropped_frames"]
        n_images = (data["dropped_frames"] == False).sum()  # noqa: E712 (works for np/torch)

        if self.training:
            packed = self.pack_actions(data["buttons"], data["j_left"], data["j_right"])
            actions, actions_mask, n_action_tokens = self._pad_action(packed)
            out["actions"] = actions
            out["actions_mask"] = actions_mask
            out["has_real_action"] = np.ones((), dtype=bool)
        else:
            n_action_tokens = self.action_horizon

        vl_token_ids, sa_token_ids = self._build_token_ids(n_images, n_action_tokens)
        vl_token_ids, vl_attn_mask = self._pad_vl_sequence(vl_token_ids)

        out["vl_token_ids"] = vl_token_ids
        out["sa_token_ids"] = sa_token_ids
        out["vl_attn_mask"] = vl_attn_mask
        out["embodiment_id"] = torch.tensor(0, dtype=torch.long)

        if self.game_mapping:
            game_name = data.get("game")
            assert game_name in self.game_mapping, f"Game '{game_name}' not in game mapping."
            out["game_ids"] = torch.tensor(self.game_mapping[game_name], dtype=torch.long)
        else:
            out["game_ids"] = torch.tensor(0, dtype=torch.long)
        return out

    def decode(self, model_output: dict) -> dict:
        j_left, j_right, buttons = self.unpack_actions(model_output["action_tensor"])
        return {"j_left": j_left, "j_right": j_right, "buttons": buttons}

    # ------------------------------------------------------------------ internals

    def _pad_action(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Zero-pad (T, real_dim) to (T, max_action_dim); mask marks real dims."""
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"
        n_action_tokens, n_dims = actions.shape
        assert n_dims <= self.max_action_dim, f"Action dim {n_dims} > max {self.max_action_dim}"

        padded = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_dims)), "constant")
        mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        mask[:, :n_dims] = True
        return padded, mask, n_action_tokens

    def _build_token_ids(self, n_images: int, n_action_tokens: int) -> tuple[np.ndarray, np.ndarray]:
        vl_token_ids = []
        if self.game_mapping:
            vl_token_ids.append(GAME_ID_TOKEN)
        for _ in range(int(n_images)):
            vl_token_ids.extend([IMG_TOKEN] * self.num_visual_tokens_per_frame)
        sa_token_ids = [ACT_TOKEN] * n_action_tokens
        return np.array(vl_token_ids), np.array(sa_token_ids)

    def _pad_vl_sequence(self, vl_token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Left-pad the VL sequence to max_sequence_length; mask is 1 on real tokens."""
        seq_len = vl_token_ids.shape[0]
        if seq_len > self.max_sequence_length:
            raise ValueError(f"VL sequence length {seq_len} exceeds max {self.max_sequence_length}")
        pad = self.max_sequence_length - seq_len
        vl_token_ids = np.pad(vl_token_ids, (pad, 0), constant_values=PAD_TOKEN)
        vl_attn_mask = np.pad(np.ones(seq_len, dtype=bool), (pad, 0), constant_values=0)
        return vl_token_ids, vl_attn_mask
