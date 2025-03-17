import gc
from pathlib import Path
from typing import Optional

import lightning as L
import rich
import torch.cuda
import wandb
from dynaconf import LazySettings
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from rich.table import Table
from torch.utils.data import DataLoader

from opengrammar.data import MiniPileCollate, MiniPileDataset
from opengrammar.model.meta import OpenGrammar


class Experiment:
    def __init__(self, settings: LazySettings):
        """
        Check if the settings are correct.

        :param settings: A settings object.
        """
        self.settings = settings

        self.wandb_logger: Optional[WandbLogger] = None
        if "wandb_token" in self.settings:
            rich.print("[[green]Token Found[/green]]: Using WandB")
            token = self.settings["wandb_token"]
            wandb.login(key=token)
            self.wandb_logger = WandbLogger(project="OpenGrammar", log_model="all")

        # Path Constants
        if "minipile_root" not in settings:
            rich.print(
                "[red]Config Not Found[/red]: "
                "`minipile_root` not in a config.toml or `OPENGRAMMAR_MINIPILE_ROOT` environment variable"
            )
            exit(1)

        minipile_root_path = Path(settings["minipile_root"])

        if not minipile_root_path.exists():
            rich.print("[red]MiniPile Dataset Path Not Found[/red]")
            exit(1)

        self.minipile_root_path = minipile_root_path

        # Hyper-Parameters
        self.batch = settings.get("batch", None)
        self.lr = settings.get("lr", 0.00001)
        self.epochs = settings("epochs", 10)
        self.hidden_dim = settings.get("hidden_dim", 128)
        self.tensor_cores = settings.get("tensor_cores", True)
        self.devices = settings.get("devices", 1)
        self.debug = settings.get("debug", False)

    def train(
        self,
        batch: Optional[int] = None,
        model_path: Optional[Path] = None,
    ) -> bool:
        """
        Train the model.

        :batch: The number of samples per batch.
        :model_path: The model path.
        :return: True if the training was successful, False otherwise.
        """

        rich.print("[[yellow]Starting Training[/yellow]]")

        # Seed
        L.seed_everything(7, verbose=False)

        # Initialize Train Dataset
        train_dataset = MiniPileDataset(root_path=self.minipile_root_path)

        # Initialize Validation Dataset
        val_dataset = MiniPileDataset(
            root_path=self.minipile_root_path, validation=True
        )

        # Override Batch Size
        if batch:
            if self.batch:
                self.batch = batch
        else:
            if not self.batch:
                self.batch = 4

        # Construct Collator
        collator = MiniPileCollate()

        # Construct DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch,
            num_workers=3,
            collate_fn=collator,
            persistent_workers=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch,
            num_workers=3,
            collate_fn=collator,
            persistent_workers=True,
        )

        table = Table(title="Hyperparameters")
        table.add_column("Batch", style="cyan", no_wrap=True)
        table.add_column("Learning Rate", style="cyan", no_wrap=True)
        table.add_column("Epochs", style="cyan", no_wrap=True)
        table.add_column("Hidden Size", style="cyan", no_wrap=True)
        table.add_row(
            str(self.batch),
            str(self.lr),
            str(self.epochs),
            str(self.hidden_dim),
        )
        rich.print(table)

        # Set Float32 Matmul Precision
        if self.tensor_cores:
            torch.set_float32_matmul_precision("high")
            rich.print("Tensor Core Precision: [green]High[/green]")

        # Load Checkpoint
        if model_path:
            model = OpenGrammar.load_from_checkpoint(
                model_path,
                settings=self.settings,
                wandb_logger=self.wandb_logger,
            )
        else:
            model = OpenGrammar(
                settings=self.settings,
                lr=self.lr,
                hidden_dim=self.hidden_dim,
                wandb_logger=self.wandb_logger,
                batch_size=self.batch,
            )

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="./checkpoints",
            filename=f"opengrammar",
            save_top_k=3,
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        rich_progress_bar = RichProgressBar()

        # Setup Trainer
        trainer = L.Trainer(
            max_epochs=self.epochs,
            callbacks=[checkpoint_callback, lr_monitor, rich_progress_bar],
            devices=self.devices,
            accelerator="gpu",
            enable_progress_bar=True,
            logger=self.wandb_logger,
            log_every_n_steps=1,
            strategy="auto",
            precision="16-mixed" if self.tensor_cores else None,
            fast_dev_run=self.debug,
            inference_mode=False,
        )

        # Start Model Training
        trainer.fit(model, train_dataloader, val_dataloader)

        # Clear Cache
        torch.cuda.empty_cache()
        _ = gc.collect()

        if "wandb_token" in self.settings:
            self.wandb_logger.experiment.finish()

        rich.print("[[green]Training Done[/green]]")
        return True
