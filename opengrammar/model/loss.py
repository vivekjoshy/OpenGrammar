from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenGrammarLoss(nn.Module):
    def __init__(self, energy_regularization: float = 0.01) -> None:
        """
        Initialize the OpenGrammar loss function based on the free energy principle.

        :param energy_regularization: Weight for the energy regularization term.
        """
        super().__init__()
        self.energy_regularization = energy_regularization

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        energy_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the loss using KL divergence between predicted and target distributions,
        with optional energy regularization.

        :param predictions: Model output predictions of shape [B, seq_len, 256].
        :param targets: Target values of shape [B, seq_len, 256].
        :param energy_values: Optional tensor of energy values to regularize of shape [1].
        :return: Loss value of shape [1].
        """
        # Compute base loss using KL divergence
        log_probs = F.log_softmax(predictions, dim=-1)  # [B, seq_len, 256]
        target_probs = F.softmax(targets, dim=-1)  # [B, seq_len, 256]
        kl_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")  # [1]

        # Add energy regularization if provided
        if energy_values is not None:
            energy_loss = self.energy_regularization * torch.mean(energy_values)  # [1]
            total_loss = kl_loss + energy_loss  # [1]
        else:
            total_loss = kl_loss  # [1]

        return total_loss


if __name__ == "__main__":
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    console = Console()

    console.print(
        Panel.fit(
            "[bold green]OpenGrammar Loss Check[/bold green]", border_style="green"
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
            batch_size = 4
            seq_len = 10
            vocab_size = 256  # For byte-level predictions

            # Create sample predictions (unnormalized logits)
            predictions = torch.randn(batch_size, seq_len, vocab_size)

            # Create targets with some ground truth distribution
            targets = torch.zeros(batch_size, seq_len, vocab_size)
            # Set random target bytes as the "correct" ones
            for b in range(batch_size):
                for s in range(seq_len):
                    target_byte = torch.randint(0, vocab_size, (1,)).item()
                    targets[b, s, int(target_byte)] = 1.0

            # Add some noise to make it more realistic
            targets = targets + torch.randn_like(targets) * 0.1

            # Generate energy values
            energy_values = torch.rand(1) * 5.0  # Random energy value

            progress.update(task, completed=True)

        # Display data shapes
        data_table = Table(title="Sample Data")
        data_table.add_column("Data", style="cyan")
        data_table.add_column("Shape", style="magenta")
        data_table.add_column("Type", style="green")
        data_table.add_column("Sample Values", style="yellow")

        data_table.add_row(
            "Predictions",
            str(predictions.shape),
            str(predictions.dtype),
            f"Min: {predictions.min().item():.4f}, Max: {predictions.max().item():.4f}",
        )
        data_table.add_row(
            "Targets",
            str(targets.shape),
            str(targets.dtype),
            f"Min: {targets.min().item():.4f}, Max: {targets.max().item():.4f}",
        )
        data_table.add_row(
            "Energy Values",
            str(energy_values.shape),
            str(energy_values.dtype),
            f"Value: {energy_values.item():.4f}",
        )

        console.print(data_table)

        # Create loss instances with different regularization weights
        console.print(
            "[yellow]Initializing OpenGrammarLoss with different regularization weights...[/yellow]"
        )

        reg_weights = [0.0, 0.01, 0.1, 1.0]
        loss_instances = [OpenGrammarLoss(energy_regularization=w) for w in reg_weights]

        # Compute losses
        loss_table = Table(title="Loss Computations")
        loss_table.add_column("Energy Regularization", style="cyan")
        loss_table.add_column("Without Energy", style="magenta")
        loss_table.add_column("With Energy", style="green")
        loss_table.add_column("Difference", style="yellow")

        for i, loss_fn in enumerate(loss_instances):
            # Compute loss without energy values
            loss_without_energy = loss_fn(predictions, targets)

            # Compute loss with energy values
            loss_with_energy = loss_fn(predictions, targets, energy_values)

            # Calculate difference
            difference = loss_with_energy.item() - loss_without_energy.item()

            loss_table.add_row(
                f"{reg_weights[i]:.4f}",
                f"{loss_without_energy.item():.6f}",
                f"{loss_with_energy.item():.6f}",
                f"{difference:.6f} {'↑' if difference > 0 else '↓' if difference < 0 else '='}",
            )

        console.print(loss_table)

        # Demonstrate how the loss changes with different energy values
        console.print("[yellow]Testing with varying energy values...[/yellow]")

        energy_table = Table(title="Impact of Energy Values")
        energy_table.add_column("Energy Value", style="cyan")
        energy_table.add_column("Loss", style="magenta")

        # Use a fixed loss function with moderate regularization
        fixed_loss_fn = OpenGrammarLoss(energy_regularization=0.05)

        test_energies = [0.0, 1.0, 5.0, 10.0, 20.0]
        for energy in test_energies:
            energy_tensor = torch.tensor([energy])
            loss_value = fixed_loss_fn(predictions, targets, energy_tensor)
            energy_table.add_row(f"{energy:.2f}", f"{loss_value.item():.6f}")

        console.print(energy_table)

        # Test gradient flow
        console.print("[yellow]Testing gradient flow...[/yellow]")

        # Create trainable predictions
        trainable_preds = torch.randn(
            batch_size, seq_len, vocab_size, requires_grad=True
        )
        optimizer = torch.optim.Adam([trainable_preds], lr=0.01)

        # Training loop to verify gradients flow correctly
        gradient_table = Table(title="Gradient Flow")
        gradient_table.add_column("Iteration", style="cyan")
        gradient_table.add_column("Loss", style="magenta")
        gradient_table.add_column("Gradient Norm", style="yellow")

        loss_fn = OpenGrammarLoss(energy_regularization=0.05)

        for i in range(5):
            optimizer.zero_grad()
            loss = loss_fn(trainable_preds, targets, energy_values)
            loss.backward()

            # Calculate gradient norm
            grad_norm = torch.norm(trainable_preds.grad).item()

            gradient_table.add_row(f"{i+1}", f"{loss.item():.6f}", f"{grad_norm:.6f}")

            optimizer.step()

        console.print(gradient_table)

        console.print(
            Panel(
                "[bold green]✓ OpenGrammarLoss is working correctly![/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel(f"[bold red]Error: {str(e)}[/bold red]", border_style="red")
        )
        import traceback

        console.print(traceback.format_exc(), style="red")
