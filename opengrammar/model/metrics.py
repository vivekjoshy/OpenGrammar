from typing import Dict, Optional, Union

import torch
import torchmetrics
from torch import Tensor


class OpenGrammarMetrics(torchmetrics.Metric):
    def __init__(self) -> None:
        """
        Initialize the OpenGrammar metrics.

        Tracks several metrics for the OpenGrammar model:
        - KL Divergence (prediction quality)
        - Energy Efficiency (energy usage)
        - Neuroplasticity (rate of weight changes)
        - Byte-level Accuracy (for tokenless predictions)
        """
        super().__init__()

        # Register state for each metric we want to track
        self.add_state("total_kl_div", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "total_plasticity", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("correct_bytes", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_bytes", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        predictions: Tensor,
        targets: Tensor,
        energy_values: Optional[Union[Tensor, float]] = None,
        plasticity_rate: Optional[Union[Tensor, float]] = None,
        byte_predictions: Optional[Tensor] = None,
        byte_targets: Optional[Tensor] = None,
    ) -> None:
        """
        Update metrics with a new batch of data.

        Shape glossary:
        - B: Batch size
        - S: Sequence length
        - V: Vocabulary size

        :param predictions: Model predictions [B, S, V]
        :param targets: Target distribution [B, S, V]
        :param energy_values: Energy usage values (optional)
        :param plasticity_rate: Neuroplasticity rate (optional)
        :param byte_predictions: Byte-level predictions [B, S, 256] (optional)
        :param byte_targets: Byte-level targets [B, S] (optional)
        """
        batch_size = predictions.size(0)
        batch_tensor = torch.tensor(batch_size, device=self.num_samples.device)
        self.num_samples += batch_tensor

        # Calculate KL divergence
        kl_div = torch.nn.functional.kl_div(
            torch.log_softmax(predictions, dim=-1),
            targets,
            reduction="batchmean",
            log_target=False,
        )
        self.total_kl_div += kl_div * batch_tensor

        # Update energy metrics if available
        if energy_values is not None:
            if not isinstance(energy_values, torch.Tensor):
                energy_tensor = torch.tensor(energy_values, device=predictions.device)
            else:
                energy_tensor = energy_values
            self.total_energy += energy_tensor.sum()

        # Update plasticity metrics if available
        if plasticity_rate is not None:
            if not isinstance(plasticity_rate, torch.Tensor):
                plasticity_tensor = torch.tensor(
                    plasticity_rate, device=predictions.device
                )
            else:
                plasticity_tensor = plasticity_rate
            self.total_plasticity += plasticity_tensor.sum()

        # Update byte accuracy metrics if available
        if byte_predictions is not None and byte_targets is not None:
            byte_preds = byte_predictions.argmax(dim=-1)  # [B, S]
            self.correct_bytes += (byte_preds == byte_targets).sum()
            bytes_tensor = torch.tensor(
                byte_targets.numel(), device=self.total_bytes.device
            )
            self.total_bytes += bytes_tensor

    def compute(self) -> Dict[str, Tensor]:
        """
        Compute final metrics.

        :return: Dictionary containing all computed metrics.
        """
        metrics: Dict[str, Tensor] = {
            "kl_divergence": self.total_kl_div / self.num_samples.float(),
        }

        if self.total_energy.item() > 0:
            metrics["energy_efficiency"] = self.total_energy / self.num_samples.float()

        if self.total_plasticity.item() > 0:
            metrics["plasticity_rate"] = (
                self.total_plasticity / self.num_samples.float()
            )

        if self.total_bytes.item() > 0:
            metrics["byte_accuracy"] = (
                self.correct_bytes.float() / self.total_bytes.float()
            )

        return metrics

    def reset(self) -> None:
        """
        Reset all metrics to zero.
        """
        device = self.device

        self.num_samples = torch.tensor(0, device=device)
        self.total_kl_div = torch.tensor(0.0, device=device)
        self.total_energy = torch.tensor(0.0, device=device)
        self.total_plasticity = torch.tensor(0.0, device=device)
        self.total_bytes = torch.tensor(0, device=device)
        self.correct_bytes = torch.tensor(0, device=device)


if __name__ == "__main__":
    import time

    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    console = Console()

    console.print(
        Panel.fit(
            "[bold green]OpenGrammar Metrics Check[/bold green]", border_style="green"
        )
    )

    try:
        # Create sample data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[yellow]Creating test data...", total=None)

            # Define dimensions
            batch_size = 5
            seq_len = 12
            vocab_size = 256  # For byte-level predictions

            # Create sample predictions and targets
            predictions = torch.randn(batch_size, seq_len, vocab_size)
            # Create soft targets (like probabilities)
            targets = torch.softmax(
                torch.randn(batch_size, seq_len, vocab_size), dim=-1
            )

            # Energy values (simulating model energy usage)
            energy_values = torch.rand(batch_size) * 10.0

            # Plasticity rate (simulating model plasticity)
            plasticity_rate = torch.rand(batch_size) * 0.1

            # Byte level data
            byte_predictions = torch.randn(batch_size, seq_len, 256)
            byte_targets = torch.randint(0, 256, (batch_size, seq_len))

            progress.update(task, completed=True)

        # Display data shapes
        data_table = Table(title="Sample Data")
        data_table.add_column("Data", style="cyan")
        data_table.add_column("Shape", style="magenta")
        data_table.add_column("Type", style="green")

        data_table.add_row(
            "Predictions", str(predictions.shape), str(predictions.dtype)
        )
        data_table.add_row("Targets", str(targets.shape), str(targets.dtype))
        data_table.add_row(
            "Energy Values", str(energy_values.shape), str(energy_values.dtype)
        )
        data_table.add_row(
            "Plasticity Rate", str(plasticity_rate.shape), str(plasticity_rate.dtype)
        )
        data_table.add_row(
            "Byte Predictions", str(byte_predictions.shape), str(byte_predictions.dtype)
        )
        data_table.add_row(
            "Byte Targets", str(byte_targets.shape), str(byte_targets.dtype)
        )

        console.print(data_table)

        # Create metrics instance
        console.print("[yellow]Initializing OpenGrammarMetrics...[/yellow]")
        metrics = OpenGrammarMetrics()

        # Update metrics with batch 1
        console.print("[yellow]Updating metrics with batch 1...[/yellow]")
        metrics.update(
            predictions=predictions,
            targets=targets,
            energy_values=energy_values,
            plasticity_rate=plasticity_rate,
            byte_predictions=byte_predictions,
            byte_targets=byte_targets,
        )

        # Show intermediate results
        results1 = metrics.compute()

        results_table1 = Table(title="Metrics After Batch 1")
        results_table1.add_column("Metric", style="cyan")
        results_table1.add_column("Value", style="magenta")

        for metric_name, metric_value in results1.items():
            results_table1.add_row(metric_name, f"{metric_value.item():.6f}")

        console.print(results_table1)

        # Simulate a second batch with different values
        console.print("[yellow]Creating and processing batch 2...[/yellow]")

        # Create more challenging second batch
        predictions2 = (
            torch.randn(batch_size, seq_len, vocab_size) * 0.5
        )  # Lower confidence
        targets2 = torch.softmax(
            torch.randn(batch_size, seq_len, vocab_size) * 2.0, dim=-1
        )  # Higher certainty

        # Different energy and plasticity profile
        energy_values2 = torch.rand(batch_size) * 15.0  # Higher energy
        plasticity_rate2 = torch.rand(batch_size) * 0.05  # Lower plasticity

        # Byte level with higher accuracy
        byte_predictions2 = torch.zeros(batch_size, seq_len, 256)
        byte_targets2 = torch.randint(0, 256, (batch_size, seq_len))

        # Make byte_predictions2 more accurate by putting high values at the correct indices
        for b in range(batch_size):
            for s in range(seq_len):
                correct_byte = byte_targets2[b, s].item()
                byte_predictions2[b, s, int(correct_byte)] = (
                    5.0  # High probability for correct byte
                )

        # Add noise
        byte_predictions2 += torch.randn_like(byte_predictions2) * 0.5

        # Update metrics with batch 2
        metrics.update(
            predictions=predictions2,
            targets=targets2,
            energy_values=energy_values2,
            plasticity_rate=plasticity_rate2,
            byte_predictions=byte_predictions2,
            byte_targets=byte_targets2,
        )

        # Show final results
        results2 = metrics.compute()

        results_table2 = Table(title="Metrics After Batch 1 & 2")
        results_table2.add_column("Metric", style="cyan")
        results_table2.add_column("Value", style="magenta")
        results_table2.add_column("Change", style="yellow")

        for metric_name, metric_value in results2.items():
            change = ""
            if metric_name in results1:
                diff = metric_value.item() - results1[metric_name].item()
                change = f"{diff:+.6f}"
                # Add arrow indicators
                if diff > 0:
                    change += " ↑"
                elif diff < 0:
                    change += " ↓"

            results_table2.add_row(metric_name, f"{metric_value.item():.6f}", change)

        console.print(results_table2)

        # Test reset functionality
        console.print("[yellow]Testing reset functionality...[/yellow]")
        metrics.reset()

        # Verify reset worked
        results_after_reset = metrics.compute()

        reset_table = Table(title="Metrics After Reset")
        reset_table.add_column("Metric", style="cyan")
        reset_table.add_column("Value", style="magenta")

        if len(results_after_reset) == 1 and "kl_divergence" in results_after_reset:
            reset_table.add_row(
                "kl_divergence", f"{results_after_reset['kl_divergence'].item():.6f}"
            )
            reset_table.add_row(
                "All other metrics", "Reset to zero (not in results dict)"
            )
            console.print(reset_table)
            console.print("[green]✓ Reset functionality working correctly![/green]")
        else:
            for metric_name, metric_value in results_after_reset.items():
                reset_table.add_row(metric_name, f"{metric_value.item():.6f}")
            console.print(reset_table)

        console.print(
            Panel(
                "[bold green]✓ OpenGrammarMetrics is working correctly![/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel(f"[bold red]Error: {str(e)}[/bold red]", border_style="red")
        )
        import traceback

        console.print(traceback.format_exc(), style="red")
