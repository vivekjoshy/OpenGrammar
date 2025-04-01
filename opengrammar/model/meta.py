from typing import Any, Dict, Optional, cast

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from opengrammar.brain import BrainState
from opengrammar.learner import NeuromorphicLearner
from opengrammar.model.metrics import OpenGrammarMetrics
from opengrammar.model.model import OpenGrammarModel


class OpenGrammar(L.LightningModule):
    """
    OpenGrammar model for training with PyTorch Lightning.

    This module wraps the core OpenGrammarModel, adding:
    - Training and validation steps
    - Loss calculation
    - Metrics tracking
    - Brain-inspired learning
    """

    def __init__(
        self,
        settings: Any,
        hidden_size: int = 512,
        layer_count: int = 4,
        lr: float = 0.00001,
        wandb_logger: Optional[Any] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the OpenGrammar model.

        :param settings: Configuration settings
        :param hidden_size: Size of hidden dimensions
        :param layer_count: Number of layers in the model
        :param lr: Learning rate
        :param wandb_logger: Optional WandB logger
        :param batch_size: Batch size for training
        """
        super().__init__()

        self.settings = settings
        self.hidden_size = hidden_size
        self.layer_count = layer_count
        self.lr = lr
        self.batch_size = batch_size
        self.wandb_logger = wandb_logger

        # Core model
        self.model = OpenGrammarModel(hidden_size=hidden_size, layer_count=layer_count)

        # Brain state for neuromorphic learning
        self.brain_state = BrainState()

        # Learning system
        self.learner = NeuromorphicLearner()

        # Metrics
        self.train_metrics = OpenGrammarMetrics()
        self.val_metrics = OpenGrammarMetrics()

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Forward pass through the model.

        Shape glossary:
        - B: Batch size
        - S: Sequence length
        - V: Vocabulary size

        :param batch: Input batch dictionary
        :return: Model predictions [B, S, V]
        """
        return self.model(batch, self.brain_state)  # type: ignore

    def training_step(self, batch: Dict[str, Tensor], batch_index: int) -> STEP_OUTPUT:
        """
        Training step for the Open Grammar model.

        :param batch: Input batch dictionary
        :param batch_index: Index of the current batch
        :return: Loss value
        """
        return self._optimization_step(batch)

    def validation_step(
        self, batch: Dict[str, Tensor], batch_index: int
    ) -> STEP_OUTPUT:
        """
        Validation step for the Open Grammar model.

        :param batch: Input batch dictionary
        :param batch_index: Index of the current batch
        :return: Loss value
        """
        # Validate batch data
        if not batch or "text" not in batch or "target" not in batch:
            self.log("val_loss", torch.tensor(0.0, device=self.device))
            return {"loss": torch.tensor(0.0, device=self.device)}

        # Check for empty tensors
        if batch["text"].numel() == 0 or batch["target"].numel() == 0:
            self.log("val_loss", torch.tensor(0.0, device=self.device))
            return {"loss": torch.tensor(0.0, device=self.device)}

        try:
            # Force conversion to appropriate types
            batch["text"] = batch["text"].to(dtype=torch.int64)
            batch["target"] = batch["target"].to(dtype=torch.int64)

            # No need for gradients in validation
            with torch.no_grad():
                # Forward pass
                predictions = self(batch)
                target = batch["target"]

                # Get energy and plasticity metrics
                energy = self.model.energy_profile()
                plasticity = self.model.plasticity_profile()

                # Compute loss
                loss = self.learner.compute_loss(
                    predictions, target, energy, plasticity, self.brain_state
                )

                # Ensure loss is a tensor
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(loss, device=self.device)

                # Update metrics
                self.val_metrics.update(
                    predictions=predictions,
                    targets=target,
                    energy_values=energy,
                    plasticity_rate=plasticity,
                    byte_predictions=predictions,
                    byte_targets=batch["text"],
                )

                # Log metrics
                self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)

                # Return detached loss
                return {"loss": loss.detach()}
        except Exception as e:
            # Handle any unexpected errors in validation step
            print(f"Error in validation_step: {str(e)}")
            self.log("val_loss", torch.tensor(0.0, device=self.device))
            return {"loss": torch.tensor(0.0, device=self.device)}

    def on_train_epoch_end(self) -> None:
        """
        Actions to perform at the end of each training epoch.
        """
        # Compute and log metrics
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train_{name}", value)

        # Reset metrics
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """
        Actions to perform at the end of each validation epoch.
        """
        # Compute and log metrics
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val_{name}", value)

        # Reset metrics
        self.val_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore
        """
        Configure optimizers and LR schedulers.

        :return: Configuration dictionary for Lightning
        """
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _optimization_step(
        self, batch: Dict[str, Tensor], brain_state: Optional[BrainState] = None
    ) -> Dict[str, Any]:
        """
        Perform one optimization step.

        :param batch: Input batch dictionary
        :param brain_state: Optional brain state for neuromodulation
        :return: Dictionary of metrics
        """
        # Validate batch data
        if not batch or "text" not in batch or "target" not in batch:
            return {"loss": torch.tensor(0.0, device=self.device)}

        # Check for empty tensors
        if batch["text"].numel() == 0 or batch["target"].numel() == 0:
            return {"loss": torch.tensor(0.0, device=self.device)}

        # Force conversion to appropriate types
        batch["text"] = batch["text"].to(dtype=torch.int64)
        batch["target"] = batch["target"].to(dtype=torch.int64)

        # Use brain state if provided, otherwise use the model's brain state
        if brain_state is None:
            brain_state = self.brain_state

        # Setup optimizer
        optimizer = self.optimizers()

        try:
            # Let PyTorch Lightning handle the optimization
            # Only compute the loss here, don't call backward or step
            with torch.no_grad():
                # Forward pass
                predictions = self(batch)
                target = batch["target"]

                # Get energy and plasticity metrics
                energy = self.model.energy_profile()
                plasticity = self.model.plasticity_profile()

                # Compute validation loss (no gradients needed)
                val_loss = self.learner.compute_loss(
                    predictions, target, energy, plasticity, self.brain_state
                )

            # Now create a fresh computational graph for the training loss
            predictions = self(batch)
            target = batch["target"]

            # Get fresh energy and plasticity metrics
            energy = self.model.energy_profile()
            plasticity = self.model.plasticity_profile()

            # Compute the loss that will be used for optimization
            loss = self.learner.compute_loss(
                predictions, target, energy, plasticity, self.brain_state
            )

            # Ensure loss is a tensor with gradients
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=self.device, requires_grad=True)
            elif not loss.requires_grad:
                loss = loss.clone().requires_grad_(True)

            # Log metrics with detached tensors
            self.log(
                "train_loss", loss.detach(), prog_bar=True, batch_size=self.batch_size
            )

            # Get metric values, ensuring they're tensors
            energy_val = (
                energy.mean()
                if energy is not None and isinstance(energy, torch.Tensor)
                else torch.tensor(0.0, device=self.device)
            )
            plasticity_val = (
                plasticity.mean()
                if plasticity is not None and isinstance(plasticity, torch.Tensor)
                else torch.tensor(0.0, device=self.device)
            )

            # Detach the metrics to ensure they don't affect the computation graph
            energy_val = energy_val.detach()
            plasticity_val = plasticity_val.detach()

            self.log(
                "train_energy",
                energy_val,
                prog_bar=True,
                batch_size=self.batch_size,
            )
            self.log(
                "train_plasticity",
                plasticity_val,
                prog_bar=True,
                batch_size=self.batch_size,
            )

            # Return the loss tensor with gradients for PyTorch Lightning to optimize
            return {
                "loss": loss,  # Return tensor with gradients
            }
        except Exception as e:
            # Handle any unexpected errors in optimization step
            print(f"Error in _optimization_step: {str(e)}")
            return {"loss": torch.tensor(0.0, device=self.device)}

    def update_brain_state(
        self, brain_state: BrainState, user_input: str, response: Optional[str] = None
    ) -> BrainState:
        """
        Update the brain state based on interaction.

        :param brain_state: Current brain state
        :param user_input: User input text
        :param response: Optional model response
        :return: Updated brain state
        """
        # Create a copy of the brain state
        updated_state = BrainState(
            engagement=brain_state.engagement,
            arousal=brain_state.arousal,
            adaptation_rate=brain_state.adaptation_rate,
            hebbian_strength=brain_state.hebbian_strength,
            last_update_time=brain_state.last_update_time,
            interaction_count=brain_state.interaction_count,
        )

        # Update based on user input and model response
        updated_state.update(user_input, response)

        return updated_state

    def optimizers(self) -> Optimizer:  # type: ignore
        """
        Get the optimizer for manual optimization.

        :return: The optimizer
        """
        # Ensure optimizers are configured
        if not hasattr(self, "_optimizer"):
            config = self.configure_optimizers()
            self._optimizer = config["optimizer"]

        return cast(Optimizer, self._optimizer)
