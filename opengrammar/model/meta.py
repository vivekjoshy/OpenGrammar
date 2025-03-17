from typing import Dict

import lightning as L
import torch
from dynaconf import LazySettings
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
)

from opengrammar.model.loss import OpenGrammarLoss
from opengrammar.model.metrics import OpenGrammarMetrics
from opengrammar.model.model import OpenGrammarModel


class OpenGrammar(L.LightningModule):
    def __init__(
        self,
        settings: LazySettings,
        lr: float = 0.00001,
        hidden_dim: int = 128,
        wandb_logger: WandbLogger = None,
        batch_size: int = 4,
    ):
        """
        The Open Grammar Model

        :param settings: A settings object.
        :param lr: The learning rate.
        :param hidden_dim: The hidden dimension of the model.
        :param wandb_logger: The wandb logger.
        :param batch_size: The batch size.
        :return: None
        """
        super(OpenGrammar, self).__init__()
        self.save_hyperparameters(ignore=["model", "settings", "wandb_logger"])
        self.settings = settings
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.wandb_logger = wandb_logger
        self.batch_size = batch_size

        # Initialize Model
        self.model = OpenGrammarModel(hidden_dim=self.hidden_dim)

        # Loss + Metrics
        self.loss = OpenGrammarLoss()
        self.train_metrics = OpenGrammarMetrics()
        self.val_metrics = OpenGrammarMetrics()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.

        :param batch: Input batch dictionary
        :return: Model output logits
        """
        return self.model(batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_index: int):
        """
        Training step for the Open Grammar model.

        :param batch: Input batch dictionary
        :param batch_index: Index of the current batch
        :return: Loss value
        """
        ...

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_index: int):
        """
        Validation step for the Open Grammar model.

        :param batch: Input batch dictionary
        :param batch_idx: Index of the current batch
        :return: Loss value
        """
        ...

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.

        :return: None
        """
        # Reset metrics after the validation epoch
        self.val_metrics.reset()

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.

        :return: None
        """
        # Reset metrics after the training epoch
        self.train_metrics.reset()

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        :return: Optimizer configuration dictionary
        """
        optimizer = AdamW(
            self.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-6
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=5,
            mode="min",
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
