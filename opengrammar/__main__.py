import gc
from pathlib import Path
from typing import Optional

import rich
import torch
from dynaconf import Dynaconf
from typer import Typer

from opengrammar import __version__
from opengrammar.experiment import Experiment

# Command Line Interface
cli: Typer = Typer()


@cli.command(help="Show Version")
def version() -> None:
    rich.print(f"Open Grammar: [blue]v{__version__}[/blue]")


@cli.command(help="Start Training")
def train(
    config: Optional[Path] = None,
    batch: Optional[int] = None,
    model: Optional[Path] = None,
) -> None:
    if config:
        # Check if config.toml
        if not str(config).endswith("config.toml"):
            rich.print("[[red]Invalid Config[/red]]: Not a config.toml file.")
            return

        if not config.exists():
            rich.print("[[red]Invalid Config[/red]]: Path does not exist.")
            return

        settings_files = [str(config.absolute())]
    else:
        settings_files = []

    # Configurations
    settings = Dynaconf(
        envvar_prefix="OPENGRAMMAR",
        settings_files=settings_files,
        environments=False,
        load_dotenv=True,
        env_switcher="OPENGRAMMAR_ENV",
    )

    # Initialize Experiment
    experiment = Experiment(settings=settings)

    # Start Training
    experiment.train(
        batch=batch,
        model_path=model,
    )


@cli.command(help="Clear CUDA Cache")
def clear() -> None:
    torch.cuda.empty_cache()
    _ = gc.collect()
    torch.cuda.empty_cache()
    _ = gc.collect()
    rich.print("[green]Cache Cleared [/green]")


if __name__ == "__main__":
    cli()
