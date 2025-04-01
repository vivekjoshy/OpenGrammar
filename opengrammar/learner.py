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


def create_zero_tensor(device: torch.device) -> torch.Tensor:
    """Helper function to create a zero tensor on the appropriate device."""
    return torch.tensor(0.0, device=device)


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
        # Remove KL divergence loss and just use MSE
        self.mse_loss = nn.MSELoss()
        # Add cross entropy loss
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

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
        # Ensure inputs are valid
        if predictions.numel() == 0 or targets.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)

        # Ensure predictions have valid shape
        if len(predictions.shape) != 3:
            print(f"Warning: predictions shape {predictions.shape} is not [B, S, V]")
            return torch.tensor(0.0, device=predictions.device)

        # Base loss calculation using cross-entropy
        try:
            if len(targets.shape) != len(predictions.shape):
                # Handle integer targets directly with cross-entropy
                # Reshape predictions and targets for cross entropy
                batch_size, seq_len, vocab_size = predictions.shape

                # Ensure targets are properly clipped and converted
                targets_flat = targets.reshape(-1).clamp(0, vocab_size - 1).long()
                predictions_flat = predictions.reshape(-1, vocab_size)

                # Use standard cross entropy which handles integer targets automatically
                base_loss: torch.Tensor = self.cross_entropy(
                    predictions_flat, targets_flat
                )
            else:
                # If targets are already distributions, use a manual calculation
                # Convert both to float to avoid type issues
                targets = targets.float()
                log_probs = torch.log_softmax(predictions.float(), dim=-1)
                target_probs = torch.softmax(targets, dim=-1)

                # Manual cross-entropy calculation
                base_loss = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))

        except Exception as e:
            print(f"Error in loss computation: {e}")
            import traceback

            traceback.print_exc()
            return torch.tensor(0.0, device=predictions.device)

        # Energy regularization
        energy_loss = torch.tensor(0.0, device=predictions.device)
        if energy_level is not None:
            energy_loss = torch.mean(energy_level) * 0.01

        # Plasticity regularization
        plasticity_loss: torch.Tensor = torch.tensor(0.0, device=predictions.device)
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
        # Validate batch data
        if not batch or "text" not in batch or "target" not in batch:
            zero_tensor = create_zero_tensor(next(model.parameters()).device)
            return zero_tensor, {"energy": zero_tensor, "plasticity": zero_tensor}

        # Check for empty tensors
        if batch["text"].numel() == 0 or batch["target"].numel() == 0:
            zero_tensor = create_zero_tensor(next(model.parameters()).device)
            return zero_tensor, {"energy": zero_tensor, "plasticity": zero_tensor}

        optimizer.zero_grad()

        target = batch["target"]  # [B, S]
        try:
            # Create fresh computation graph by running forward pass
            predictions = model(batch)  # [B, S, V]

            # Get energy and plasticity metrics
            energy_level = model.energy_profile()
            plasticity_level = model.plasticity_profile()

            # Compute loss with fresh tensors
            loss = self.compute_loss(
                predictions, target, energy_level, plasticity_level, brain_state
            )

            # Ensure loss is a tensor and not a float
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(
                    loss, device=next(model.parameters()).device, requires_grad=True
                )
            elif not loss.requires_grad:
                # If the tensor doesn't require gradients, create a new tensor that does
                loss = loss.clone().detach().requires_grad_(True)

            # Perform backward pass only once
            loss.backward()  # type: ignore[no-untyped-call]

            if brain_state is not None:
                brain_state.modulate_gradients(model.named_parameters())

            optimizer.step()

            # Create metrics with detached tensors to avoid holding references to the computation graph
            metrics_loss = loss.detach().clone()
            zero_tensor = torch.tensor(0.0, device=loss.device)

            # Ensure all metrics are tensors and detached from computation graph
            metrics = {
                "loss": metrics_loss,
                "energy": (
                    energy_level.mean().detach().clone()
                    if energy_level is not None
                    and isinstance(energy_level, torch.Tensor)
                    else zero_tensor
                ),
                "plasticity": (
                    plasticity_level.mean().detach().clone()
                    if plasticity_level is not None
                    and isinstance(plasticity_level, torch.Tensor)
                    else zero_tensor
                ),
            }

            # Apply Hebbian learning with detached predictions
            if hasattr(model, "apply_hebbian") and brain_state is not None:
                hebbian_scaling = brain_state.hebbian_strength
                if callable(getattr(model, "apply_hebbian")):
                    getattr(model, "apply_hebbian")(
                        batch["text"],
                        predictions.detach().clone(),
                        scale=hebbian_scaling,
                    )

            # Return a fresh loss tensor not connected to the computation graph
            return metrics_loss, metrics

        except Exception as e:
            # Handle any unexpected errors in forward pass
            print(f"Error during optimize_step: {str(e)}")
            zero_tensor = create_zero_tensor(next(model.parameters()).device)
            return zero_tensor, {"energy": zero_tensor, "plasticity": zero_tensor}

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
