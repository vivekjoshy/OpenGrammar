"""
OpenGrammar: A brain-inspired, tokenless neural language model.

This package implements a neural language model based on
brain-inspired mechanisms including neuroplasticity, energy
constraints, and neuromodulation.
"""

from opengrammar.brain import BrainState, NeuromodulationType
from opengrammar.data import MiniPileCollate, MiniPileDataset
from opengrammar.experiment import Experiment
from opengrammar.learner import NeuromorphicLearner
from opengrammar.model.meta import OpenGrammar

# Core components
from opengrammar.model.model import OpenGrammarModel

# Metadata
__version__ = "0.1.0"
__author__ = "OpenGrammar Contributors"
__email__ = "inbox@vivekjoshy.com"
__copyright__ = "Copyright 2025, Vivek Joshy"
__deprecated__ = False
__license__ = "MIT"
__maintainer__ = "Vivek Joshy"
__status__ = "Planning"

# Public API
__all__ = [
    # Models
    "OpenGrammarModel",
    "OpenGrammar",
    "MiniPileDataset",
    "MiniPileCollate",
    "Experiment",
    "BrainState",
    "NeuromodulationType",
    "NeuromorphicLearner",
]

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    # Display package information
    console.print(
        Panel.fit(
            f"[bold green]OpenGrammar Package[/bold green]\n\n"
            f"Version: {__version__}\n"
            f"Author: {__author__}\n"
            f"Status: {__status__}\n\n"
            f"A framework for in-silico general intelligence.",
            title="Package Information",
            border_style="green",
        )
    )
