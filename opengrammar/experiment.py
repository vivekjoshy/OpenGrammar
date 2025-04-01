import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import rich
import torch
import torch.cuda
from dynaconf import LazySettings
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from rich.table import Table
from torch.utils.data import DataLoader

import wandb
from opengrammar.data import MiniPileCollate, MiniPileDataset
from opengrammar.model.meta import OpenGrammar


class Experiment:
    def __init__(self, settings: LazySettings) -> None:
        """
        Initialize the OpenGrammar experiment with settings.

        :param settings: A settings object containing configuration parameters.
        """
        self.settings = settings

        self.wandb_logger: Optional[WandbLogger] = None
        if "wandb_token" in self.settings:
            rich.print("[Token Found]: Using WandB")
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
        self.hidden_size = settings.get("hidden_size", 128)
        self.layer_count = settings.get("layer_count", 4)
        self.tensor_cores = settings.get("tensor_cores", True)
        self.devices = settings.get("devices", 1)
        self.random_seed = settings.get("random_seed", 7)
        self.debug = settings.get("debug", False)
        self.use_gpu = settings.get("use_gpu", True)
        self.checkpoint_dir = Path(settings.get("checkpoint_dir", "checkpoints"))
        self.sequence_length = settings.get("sequence_length", 1024)

        # These will be initialized when needed
        self.train_dataset: Optional[MiniPileDataset] = None
        self.val_dataset: Optional[MiniPileDataset] = None

    def train(
        self,
        batch: Optional[int] = None,
        model_path: Optional[Path] = None,
    ) -> bool:
        """
        Train the OpenGrammar model.

        :param batch: The number of samples per batch.
        :param model_path: The model checkpoint path for resuming training.
        :return: True if the training was successful, False otherwise.
        """

        rich.print("[Starting Training]")

        # Set random seed for reproducibility
        rich.print(f"Setting random seed to: {self.random_seed}")
        L.seed_everything(self.random_seed, workers=True)
        torch.manual_seed(self.random_seed)  # Set PyTorch seed explicitly

        # Initialize Train Dataset with random seed
        self.train_dataset = MiniPileDataset(
            root_path=self.minipile_root_path,
            random_seed=self.random_seed,
            sequence_length=self.sequence_length,
        )

        # Initialize Validation Dataset with random seed
        self.val_dataset = MiniPileDataset(
            root_path=self.minipile_root_path,
            validation=True,
            random_seed=self.random_seed,
            sequence_length=self.sequence_length,
        )

        # Get dataloaders
        train_dataloader, val_dataloader = self._get_dataloaders()

        # Display hyperparameters
        self._display_hyperparameters(batch)

        # Load or create model
        model = self._initialize_model(model_path, batch)

        # Configure callbacks
        callbacks = self._setup_callbacks()

        # Select training device
        accelerator = "gpu" if self.use_gpu and torch.cuda.is_available() else "cpu"

        # Configure trainer
        trainer = L.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            logger=self.wandb_logger,
            accelerator=accelerator,
            devices=self.devices,
            deterministic=True if self.random_seed is not None else False,
            precision="16-mixed" if self.tensor_cores else "32",
        )

        # Train model
        trainer.fit(model, train_dataloader, val_dataloader)

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        rich.print("[Training Done]")
        return True

    def _get_dataloaders(
        self,
    ) -> Tuple[
        DataLoader[Dict[str, torch.Tensor]], DataLoader[Dict[str, torch.Tensor]]
    ]:
        """
        Get DataLoaders for training and validation.

        :return: Tuple of (train_dataloader, val_dataloader)
        """
        if self.train_dataset is None or self.val_dataset is None:
            raise ValueError("Datasets must be initialized before creating dataloaders")

        # Initialize collate function
        collate_fn = MiniPileCollate(self.sequence_length)

        # Define worker init function with proper type annotation
        def worker_init_fn(worker_id: int) -> None:
            """Initialize worker with reproducible seed"""
            torch.manual_seed(self.random_seed + worker_id)

        # Initialize train dataloader
        train_batch_size = self.batch if self.batch is not None else 4
        train_dataloader = DataLoader[Dict[str, torch.Tensor]](
            self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        # Initialize validation dataloader
        val_dataloader = DataLoader[Dict[str, torch.Tensor]](
            self.val_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=2,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        return train_dataloader, val_dataloader

    def _display_hyperparameters(self, batch: Optional[int] = None) -> None:
        """
        Display hyperparameters in a rich table format.

        :param batch: Optional batch size to override settings
        """
        # Print hyperparameters
        table = Table(title="Hyperparameters")
        table.add_column("Batch", style="cyan", no_wrap=True)
        table.add_column("Learning Rate", style="cyan", no_wrap=True)
        table.add_column("Epochs", style="cyan", no_wrap=True)
        table.add_column("Hidden Size", style="cyan", no_wrap=True)
        table.add_column("Layer Count", style="cyan", no_wrap=True)
        table.add_column("Random Seed", style="cyan", no_wrap=True)

        # Use provided batch if available
        batch_size = batch if batch is not None else self.batch

        table.add_row(
            str(batch_size),
            str(self.lr),
            str(self.epochs),
            str(self.hidden_size),
            str(self.layer_count),
            str(self.random_seed),
        )
        rich.print(table)

        # Set Float32 Matmul Precision
        if self.tensor_cores:
            torch.set_float32_matmul_precision("high")
            rich.print("Tensor Core Precision: High")

        # Set deterministic mode for reproducibility if needed
        if self.random_seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            rich.print("CUDNN: Deterministic mode enabled for reproducibility")

    def _initialize_model(
        self, model_path: Optional[Path], batch: Optional[int]
    ) -> OpenGrammar:
        """
        Initialize the model, either from a checkpoint or create a new one.

        :param model_path: Path to an existing model checkpoint
        :param batch: Optional batch size override
        :return: Initialized OpenGrammar model
        """
        # Get batch size from parameters or config
        batch_size = batch if batch is not None else self.batch

        # Load Checkpoint
        if model_path:
            model = OpenGrammar.load_from_checkpoint(
                str(model_path),
                settings=self.settings,
                wandb_logger=self.wandb_logger,
            )
        else:
            model = OpenGrammar(
                settings=self.settings,
                hidden_size=self.hidden_size,
                layer_count=self.layer_count,
                lr=self.lr,
                wandb_logger=self.wandb_logger,
                batch_size=batch_size,
            )

        return model

    def _setup_callbacks(self) -> List[Any]:
        """
        Set up Lightning callbacks for training.

        :return: List of callback objects
        """
        # Make sure the checkpoint directory exists
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=str(self.checkpoint_dir),
            filename="opengrammar-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            mode="min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        rich_progress = RichProgressBar()

        return [checkpoint_callback, lr_monitor, rich_progress]
