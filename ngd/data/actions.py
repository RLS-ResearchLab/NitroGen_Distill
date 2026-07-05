"""Schema and readers for the NitroGen HF dataset (nvidia/NitroGen, actions-only),
plus the dataset -> model action-space conversion.

Dataset layout (per 20-second chunk):
    actions/SHARD_xxxx/<video_id>/<video_id>_chunk_xxxx/
        actions_raw.parquet         per-frame gamepad state
        actions_processed.parquet   same, after quality filtering / remapping (optional)
        metadata.json               timestamps, source url, game, crop bboxes, ...

Each parquet row is one video frame: 17 boolean button columns plus ``j_left`` /
``j_right`` columns holding [x, y] pairs in [-1, 1] ((-1, -1) = top-left).

Action spaces -- two different things:
    - The *dataset* labels 17 physical buttons (DATASET_BUTTONS below).
    - The *model* predicts 21 buttons (MODEL_BUTTONS): the 17 physical ones plus four
      virtual RIGHT_UP/RIGHT_BOTTOM/RIGHT_LEFT/RIGHT_RIGHT "buttons" for discretized
      right-stick directions. Model action vector = [21 buttons, j_left(2), j_right(2)]
      = 25 dims, no padding (verified against scripts/play.py in the official repo,
      which asserts the decoded button vector has exactly len(BUTTON_ACTION_TOKENS)=21).

``dataset_to_model_actions`` bridges the two. CAVEAT: the exact rule NVIDIA used to
derive the four virtual right-stick buttons for training labels is unpublished; the
threshold here is a guess. Validate against the teacher with open-loop probing
(ngd/eval) before trusting a training pipeline built on it.
"""

import json
from pathlib import Path

import numpy as np

# The 17 physical buttons, in dataset-card column order.
DATASET_BUTTONS = [
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "dpad_up",
    "left_shoulder",
    "left_thumb",
    "left_trigger",
    "right_shoulder",
    "right_thumb",
    "right_trigger",
    "south",
    "west",
    "east",
    "north",
    "back",
    "start",
    "guide",
]

# The 21 buttons of the model's action space, in the exact (alphabetical) order used by
# the official inference stack (nitrogen/shared.py: BUTTON_ACTION_TOKENS).
MODEL_BUTTONS = [
    "BACK",
    "DPAD_DOWN",
    "DPAD_LEFT",
    "DPAD_RIGHT",
    "DPAD_UP",
    "EAST",
    "GUIDE",
    "LEFT_SHOULDER",
    "LEFT_THUMB",
    "LEFT_TRIGGER",
    "NORTH",
    "RIGHT_BOTTOM",  # virtual: right stick pushed down
    "RIGHT_LEFT",    # virtual: right stick pushed left
    "RIGHT_RIGHT",   # virtual: right stick pushed right
    "RIGHT_SHOULDER",
    "RIGHT_THUMB",
    "RIGHT_TRIGGER",
    "RIGHT_UP",      # virtual: right stick pushed up
    "SOUTH",
    "START",
    "WEST",
]

BUTTONS = DATASET_BUTTONS  # backward-friendly alias for the raw label schema


def load_chunk_actions(chunk_dir: str | Path, prefer_processed: bool = True) -> dict[str, np.ndarray]:
    """Read one chunk's parquet into arrays: buttons (T, 17) f32, j_left/j_right (T, 2) f32.

    Buttons follow DATASET_BUTTONS order (raw label space, not the model's).
    """
    import polars as pl

    chunk_dir = Path(chunk_dir)
    path = chunk_dir / "actions_processed.parquet"
    if not (prefer_processed and path.exists()):
        path = chunk_dir / "actions_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No actions parquet in {chunk_dir}")

    df = pl.read_parquet(path)
    buttons = np.stack([df[c].cast(pl.Float32).to_numpy() for c in DATASET_BUTTONS], axis=-1)
    j_left = np.stack(df["j_left"].to_numpy()).astype(np.float32)
    j_right = np.stack(df["j_right"].to_numpy()).astype(np.float32)
    return {"buttons": buttons, "j_left": j_left, "j_right": j_right}


def dataset_to_model_actions(
    buttons: np.ndarray,  # (T, 17) in DATASET_BUTTONS order
    j_right: np.ndarray,  # (T, 2) in [-1, 1], (-1, -1) = top-left
    right_stick_threshold: float = 0.5,  # UNVERIFIED, see module docstring
) -> np.ndarray:
    """(T, 17) physical buttons + right stick -> (T, 21) model-order button vector."""
    T = buttons.shape[0]
    col = {name: buttons[:, i] for i, name in enumerate(DATASET_BUTTONS)}

    x, y = j_right[:, 0], j_right[:, 1]
    virtual = {
        "RIGHT_UP": (y < -right_stick_threshold).astype(np.float32),
        "RIGHT_BOTTOM": (y > right_stick_threshold).astype(np.float32),
        "RIGHT_LEFT": (x < -right_stick_threshold).astype(np.float32),
        "RIGHT_RIGHT": (x > right_stick_threshold).astype(np.float32),
    }

    out = np.zeros((T, len(MODEL_BUTTONS)), dtype=np.float32)
    for i, name in enumerate(MODEL_BUTTONS):
        out[:, i] = virtual[name] if name in virtual else col[name.lower()]
    return out


def load_chunk_metadata(chunk_dir: str | Path) -> dict:
    with open(Path(chunk_dir) / "metadata.json") as f:
        return json.load(f)


def iter_chunk_dirs(actions_root: str | Path):
    """Yield every chunk directory under an ``actions/`` tree (any shard)."""
    actions_root = Path(actions_root)
    for meta in sorted(actions_root.rglob("metadata.json")):
        yield meta.parent
