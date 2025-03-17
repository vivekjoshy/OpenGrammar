"""
Command-line interface for the OpenGrammar neural language model.
"""

import gc
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
from dynaconf import Dynaconf
from rich import print

from opengrammar import __version__
from opengrammar.experiment import Experiment

# Create CLI
cli = typer.Typer(
    name="opengrammar",
    help="OpenGrammar - Neuromorphic language learning",
    add_completion=False,
)


@cli.command(name="train")
def train(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    batch: Optional[int] = typer.Option(
        None, "--batch", "-b", help="Override batch size from configuration"
    ),
    model: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to existing model for continued training"
    ),
) -> None:
    """Train an OpenGrammar model."""
    try:
        # Load settings from config file if provided
        if config:
            if not config.exists():
                print(f"[red]Config file not found: {config}[/red]")
                sys.exit(1)
            settings = Dynaconf(settings_files=[config])
        else:
            # Load default settings
            settings = Dynaconf(
                settings_files=["config.toml", ".secrets.toml"],
                environments=True,
                env_prefix="OPENGRAMMAR",
                load_dotenv=True,
            )

        # Create experiment
        experiment = Experiment(settings=settings)

        # Start training
        experiment.train(
            batch=batch,
            model_path=model,
        )

    except Exception as e:
        print(f"[red]Error during training: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


@cli.command(help="Show Version")
def version() -> None:
    """Display the package version."""
    typer.echo(f"OpenGrammar: v{__version__}")


@cli.command(help="Clear CUDA Cache")
def clear() -> None:
    """Clear CUDA cache to free GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        typer.echo("GPU cache cleared")
    else:
        typer.echo("No GPU available")
