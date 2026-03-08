"""Problem interface for shakeback.

Users implement a Problem to define their model, data, and loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Problem(ABC):
    """Defines how to load, train, evaluate, and save a model.

    Subclass this and pass it to ``shakeback()`` or point the CLI at a
    module that contains a ``problem`` instance.
    """

    @abstractmethod
    def load_checkpoint(self, path: str, device: torch.device) -> tuple[nn.Module, dict]:
        """Load a checkpoint and return ``(model, checkpoint_dict)``.

        The model should already be on *device*.  The checkpoint dict is
        an opaque bag that gets passed back to :meth:`save_checkpoint`
        and :meth:`make_loader` — stash anything you need in there
        (config, tokenizer info, etc.).
        """

    @abstractmethod
    def make_loader(self, checkpoint_dict: dict, batch_size: int,
                    device: torch.device) -> DataLoader:
        """Build a DataLoader for training and evaluation."""

    @abstractmethod
    def compute_loss(self, model: nn.Module, batch: Any,
                     device: torch.device) -> torch.Tensor:
        """Run a forward pass and return a scalar loss tensor."""

    @abstractmethod
    def save_checkpoint(self, path: str, model: nn.Module,
                        checkpoint_dict: dict, extra: dict) -> None:
        """Save a checkpoint.

        *extra* contains ``val_loss``, ``total_epochs``, and
        ``total_shakes`` from shakeback — persist them however you like.
        """
