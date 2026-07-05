"""Distillation strategy interface. Implementations plug in here (see ROADMAP.md).

Design: a strategy owns the *loss computation*, not the training loop. The (future)
trainer handles optimizers, schedules, checkpointing, and logging; a strategy receives
a frozen teacher, a trainable student, and per-batch data, and returns a dict of losses.
That keeps step distillation, feature distillation, and output distillation mixable
(losses from several strategies can be summed).

Useful teacher hooks already available in ngd.model:
    - NitroGenModel.forward returns the flow-matching loss; the internals expose
      per-timestep velocity via `_predict_velocity` (teacher velocity targets).
    - DiT.forward(..., return_all_hidden_states=True) exposes every block's hidden
      states (feature-matching targets).
    - NitroGenModel.get_action(num_inference_timesteps=N) overrides the step count
      (trajectory generation for consistency / reflow style data).
"""

from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import nn

_REGISTRY: dict[str, type["DistillStrategy"]] = {}


def register_strategy(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_strategy(name: str) -> type["DistillStrategy"]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown distillation strategy '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


class DistillStrategy(ABC):
    """Base class: computes distillation losses for one batch."""

    name: str = "base"

    def __init__(self, teacher: nn.Module, student: nn.Module):
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = student

    @abstractmethod
    def loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """Return {"loss": total, <component losses for logging>...} for one batch."""
        raise NotImplementedError
