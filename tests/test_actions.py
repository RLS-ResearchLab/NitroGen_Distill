"""Dataset -> model action-space conversion."""

import numpy as np

from ngd.data.actions import DATASET_BUTTONS, MODEL_BUTTONS, dataset_to_model_actions


def test_model_button_order_is_official():
    # Must match nitrogen/shared.py BUTTON_ACTION_TOKENS exactly (alphabetical).
    assert MODEL_BUTTONS == sorted(MODEL_BUTTONS)
    assert len(MODEL_BUTTONS) == 21
    assert len(DATASET_BUTTONS) == 17
    virtual = set(MODEL_BUTTONS) - {b.upper() for b in DATASET_BUTTONS}
    assert virtual == {"RIGHT_UP", "RIGHT_BOTTOM", "RIGHT_LEFT", "RIGHT_RIGHT"}


def test_physical_buttons_pass_through():
    T = 4
    buttons = np.zeros((T, 17), dtype=np.float32)
    buttons[:, DATASET_BUTTONS.index("south")] = 1.0
    j_right = np.zeros((T, 2), dtype=np.float32)

    out = dataset_to_model_actions(buttons, j_right)
    assert out.shape == (T, 21)
    assert (out[:, MODEL_BUTTONS.index("SOUTH")] == 1).all()
    assert out.sum() == T  # nothing else pressed


def test_virtual_right_stick_buttons():
    j_right = np.array(
        [
            [0.0, -0.9],  # up (y toward top = -1)
            [0.0, 0.9],   # down
            [-0.9, 0.0],  # left
            [0.9, 0.0],   # right
            [0.1, 0.1],   # inside deadzone: nothing
        ],
        dtype=np.float32,
    )
    buttons = np.zeros((5, 17), dtype=np.float32)
    out = dataset_to_model_actions(buttons, j_right, right_stick_threshold=0.5)

    def pressed(row):
        return {MODEL_BUTTONS[i] for i in np.flatnonzero(out[row])}

    assert pressed(0) == {"RIGHT_UP"}
    assert pressed(1) == {"RIGHT_BOTTOM"}
    assert pressed(2) == {"RIGHT_LEFT"}
    assert pressed(3) == {"RIGHT_RIGHT"}
    assert pressed(4) == set()
