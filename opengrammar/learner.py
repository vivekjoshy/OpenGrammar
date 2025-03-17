"""
Implementation of brain-inspired learning algorithms for OpenGrammar.

This module contains learning algorithms inspired by neuromorphic principles,
including Hebbian learning, self-organization, and energy-based dynamics.
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from opengrammar.brain import BrainState
from opengrammar.model.model import OpenGrammarModel


class LossType(enum.Enum):
    """Types of loss functions used in neuromorphic learning."""

    PREDICTION = "prediction"  # Prediction accuracy loss
    ENERGY = "energy"  # Energy constraint loss
    PLASTICITY = "plasticity"  # Plasticity regulation loss


@dataclass
class LearningMetrics:
    """Container for learning-related metrics."""

    loss: float = 0.0
    energy: float = 0.0
    plasticity: float = 0.0


class BaseLearner(ABC):
    """
    Abstract base class for learning algorithms in OpenGrammar.

    This defines the interface that all learning algorithms must implement,
    allowing for different approaches to learning and optimization.
    """

    @abstractmethod
    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        energy: Optional[Tensor] = None,
        plasticity: Optional[Tensor] = None,
        brain_state: Optional[BrainState] = None,
    ) -> Tensor:
        """
        Compute loss for the learning algorithm.

        :param predictions: Model predictions
        :param targets: Target values
        :param energy: Energy level tensor
        :param plasticity: Plasticity level tensor
        :param brain_state: Brain state for neuromodulation
        :return: Loss tensor
        """
        pass

    @abstractmethod
    def optimize_step(
        self,
        model: OpenGrammarModel,
        optimizer: optim.Optimizer,
        batch: Dict[str, Tensor],
        brain_state: Optional[BrainState] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Perform one optimization step.

        :param model: The model to optimize
        :param optimizer: The optimizer to use
        :param batch: Batch of data
        :param brain_state: Brain state for neuromodulation
        :return: Tuple of (loss, metrics)
        """
        pass


class NeuromorphicLearner(BaseLearner):
    """
    Implements brain-inspired learning algorithms for continuous adaptation.

    This class handles specialized learning rules for neuromorphic computing,
    including Hebbian updates and neuromodulation effects.
    """

    def __init__(self) -> None:
        """Initialize the learning system with default loss functions."""
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

    def compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        energy_level: Optional[Tensor] = None,
        plasticity_level: Optional[Tensor] = None,
        brain_state: Optional[BrainState] = None,
    ) -> Tensor:
        """
        Compute loss with neuromorphic components.

        This combines standard loss functions with energy regularization
        and plasticity to create a brain-inspired learning dynamic.

        Shape glossary:
        - B: Batch size
        - S: Sequence length
        - V: Vocabulary size

        :param predictions: Model predictions [B, S, V]
        :param targets: Target values [B, S, V] or [B, S] for integer targets
        :param energy_level: Energy consumption level
        :param plasticity_level: Plasticity level
        :param brain_state: Optional brain state for neuromodulation
        :return: Combined loss value
        """
        # Base loss calculation
        if len(targets.shape) != len(predictions.shape):
            # Handle integer targets (convert to one-hot)
            batch_size, seq_len = targets.shape
            vocab_size = predictions.shape[-1]

            flat_targets = targets.reshape(-1)
            one_hot_targets = torch.zeros(
                flat_targets.size(0),
                vocab_size,
                device=targets.device,
                dtype=torch.float32,
            )
            one_hot_targets.scatter_(1, flat_targets.unsqueeze(1), 1)
            one_hot_targets = one_hot_targets.reshape(batch_size, seq_len, vocab_size)

            log_probs = torch.log_softmax(predictions, dim=-1)
            base_loss = -torch.mean(torch.sum(one_hot_targets * log_probs, dim=-1))
        else:
            # Handle distribution targets
            log_probs = torch.log_softmax(predictions, dim=-1)
            target_probs = torch.softmax(targets, dim=-1)
            base_loss = self.kl_div_loss(log_probs, target_probs)

        # Energy regularization
        energy_loss = torch.tensor(0.0, device=predictions.device)
        if energy_level is not None:
            energy_loss = torch.mean(energy_level) * 0.01

        # Plasticity regularization
        plasticity_loss = torch.tensor(0.0, device=predictions.device)
        if plasticity_level is not None:
            if brain_state is not None:
                target_plasticity = 0.5 + 0.5 * brain_state.engagement
                plasticity_loss = (
                    self.mse_loss(
                        plasticity_level.mean(),
                        torch.tensor(target_plasticity, device=plasticity_level.device),
                    )
                    * 0.005
                )
            else:
                plasticity_loss = (
                    self.mse_loss(
                        plasticity_level.mean(),
                        torch.tensor(0.5, device=plasticity_level.device),
                    )
                    * 0.005
                )

        return base_loss + energy_loss + plasticity_loss

    def optimize_step(
        self,
        model: OpenGrammarModel,
        optimizer: optim.Optimizer,
        batch: Dict[str, Tensor],
        brain_state: Optional[BrainState] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Perform one optimization step with neuromorphic principles.

        This handles the complete optimization cycle including forward pass,
        loss computation, gradient calculation, and parameter updates.

        Shape glossary:
        - B: Batch size
        - S: Sequence length
        - V: Vocabulary size

        :param model: The neuromorphic model
        :param optimizer: The optimizer
        :param batch: Batch with 'text' and 'target' keys
        :param brain_state: Optional brain state for neuromodulation
        :return: Tuple of (loss, metrics dictionary)
        """
        optimizer.zero_grad()

        target = batch["target"]  # [B, S]
        predictions = model(batch)  # [B, S, V]

        energy_level = model.energy_profile()
        plasticity_level = model.plasticity_profile()

        loss = self.compute_loss(
            predictions, target, energy_level, plasticity_level, brain_state
        )

        loss.backward()  # type: ignore[no-untyped-call]

        if brain_state is not None:
            brain_state.modulate_gradients(model.named_parameters())

        optimizer.step()

        zero_tensor = torch.tensor(0.0, device=loss.device)

        metrics = {
            "loss": loss.detach(),
            "energy": (
                energy_level.mean().detach()
                if energy_level is not None
                else zero_tensor
            ),
            "plasticity": (
                plasticity_level.mean().detach()
                if plasticity_level is not None
                else zero_tensor
            ),
        }

        if hasattr(model, "apply_hebbian") and brain_state is not None:
            hebbian_scaling = brain_state.hebbian_strength
            if callable(getattr(model, "apply_hebbian")):
                getattr(model, "apply_hebbian")(
                    batch["text"], predictions.detach(), scale=hebbian_scaling
                )

        return loss, metrics

    def create_target_distribution(
        self, observed_token: int, device: torch.device, smoothing: float = 0.1
    ) -> Tensor:
        """
        Create a smoothed target distribution for a token.

        :param observed_token: The token that was observed
        :param device: PyTorch device
        :param smoothing: Label smoothing factor
        :return: Smoothed one-hot vector of shape [V]
        """
        smoothing_value = smoothing / 255.0
        distribution = torch.full((256,), smoothing_value, device=device)
        distribution[observed_token] = 1.0 - smoothing

        return distribution


def create_zero_tensor(device: Optional[torch.device] = None) -> Tensor:
    """Helper function to create a zero tensor with proper typing."""
    return torch.tensor(0.0, device=device)
