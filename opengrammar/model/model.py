"""
Core neural network architecture for OpenGrammar.

This module implements the brain-inspired neural network model,
with neuroplasticity and energy constraints built in.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opengrammar.brain import BrainState


class NeurotransmitterModule(nn.Module):
    """
    Neural module that simulates neuromodulatory mechanisms.

    This module simulates dopamine and norepinephrine effects on neural
    processing, allowing context-dependent adaptation.
    """

    def __init__(self, hidden_size: int):
        """
        Initialize neuromodulatory module.

        :param hidden_size: Size of the hidden layer
        """
        super().__init__()
        # Base neuromodulatory pathways
        self.dopamine_gate = nn.Linear(hidden_size, hidden_size)
        self.norepinephrine_filter = nn.Linear(hidden_size, hidden_size)

        # Context adaptation layers
        self.context_modulation = nn.Linear(
            hidden_size, 2
        )  # 2 outputs: dopamine, norepinephrine

    def forward(
        self, x: torch.Tensor, brain_state: Optional[BrainState] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply neuromodulatory gating to input.

        Shape glossary:
        - B: Batch size
        - S: Sequence length
        - H: Hidden dimension size

        :param x: Input tensor [B, S, H]
        :param brain_state: Optional BrainState for modulation
        :return: Modulated tensor and neuromodulatory values
        """
        # Extract context signals from input
        context_signals = torch.mean(x, dim=1)  # [B, H]
        neuromod = self.context_modulation(context_signals)  # [B, 2]

        # Split into individual neurotransmitters
        dopamine, norepinephrine = torch.chunk(neuromod, 2, dim=1)

        # Apply brain state modulation if available
        if brain_state is not None:
            dopamine = dopamine * (0.5 + 0.5 * brain_state.engagement)
            norepinephrine = norepinephrine * (0.5 + 0.5 * brain_state.arousal)

        # Apply dopamine gate (modulates goal-directed behavior)
        x_gate = torch.sigmoid(self.dopamine_gate(x))
        x = x * (1.0 + dopamine.unsqueeze(1) * x_gate)

        # Apply norepinephrine filter (modulates signal-to-noise)
        x_noise = torch.tanh(self.norepinephrine_filter(x))
        x = x + (norepinephrine.unsqueeze(1) * x_noise)

        # Return modulated signal and modulatory values
        modulation = {
            "dopamine": dopamine.mean().detach(),
            "norepinephrine": norepinephrine.mean().detach(),
        }

        return x, modulation


class GainControlAttention(nn.Module):
    """
    Attention mechanism with gain control inspired by visual cortex.

    This attention mechanism models the brain's ability to dynamically
    adjust the gain of neural signals based on context.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4):
        """
        Initialize gain control attention.

        :param hidden_size: Size of hidden layer
        :param num_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Attention components
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.context_projection = nn.Linear(hidden_size, hidden_size)

        # Gain control component
        self.gain_control = nn.Linear(hidden_size, num_heads)

        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, x: torch.Tensor, brain_state: Optional[BrainState] = None
    ) -> torch.Tensor:
        """
        Forward pass for the attention mechanism.

        Shape glossary:
        - B: Batch size
        - S: Sequence length
        - H: Hidden dimension size
        - N: Number of attention heads
        - D: Head dimension size (H/N)

        :param x: Input tensor [B, S, H]
        :param brain_state: Optional brain state for modulation
        :return: Output tensor [B, S, H]
        """
        batch_size, seq_len, _ = x.size()

        # Split into multiple attention heads
        query = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, N, S, D]
        key = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, N, S, D]
        value = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, N, S, D]

        # Dynamic gain control - amplifies important signals
        gain = self.gain_control(x.mean(dim=1, keepdim=True))  # [B, 1, N]
        gain = gain.view(batch_size, self.num_heads, 1, 1)  # [B, N, 1, 1]
        query = query * (1.0 + gain)  # [B, N, S, D]

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [B, N, S, S]
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        # Apply attention
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [B, N, S, S]
        context = torch.matmul(attention_weights, value)  # [B, N, S, D]

        # Reshape and project
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )  # [B, S, H]
        output: torch.Tensor = self.output_projection(context)  # [B, S, H]

        return output


class EnergyConstrainedLayer(nn.Module):
    """
    Neural layer with built-in energy constraints.

    This layer models biological energy constraints in neural computation,
    encouraging efficient processing.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize energy-constrained layer.

        :param input_size: Size of input features
        :param hidden_size: Size of hidden layer
        """
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.activation = (
            nn.SiLU()
        )  # Sigmoid-weighted Linear Unit (like brain's activation)

        # Energy tracking parameters
        self.energy_factor = nn.Parameter(torch.ones(1) * 0.5)
        self.register_buffer("energy_usage", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply energy-constrained processing.

        :param x: Input tensor
        :return: Activated output tensor
        """
        # Standard forward pass
        x = self.linear(x)
        x = self.activation(x)

        # Track energy usage (simplified model of metabolic cost)
        activation_sum = torch.abs(x).sum().detach()
        weight_magnitude = torch.abs(self.linear.weight).sum().detach()

        # Energy model: combination of activation cost and weight cost
        energy_usage = (
            activation_sum * 0.5 + weight_magnitude * 0.5
        ) * self.energy_factor

        # Store for metrics
        self.energy_usage = energy_usage.detach() / (x.shape[0] * x.shape[1])

        return x

    def get_energy_usage(self) -> torch.Tensor:
        """
        Get current energy usage metric.

        :return: Energy usage tensor
        """
        return self.energy_usage


class NeuroplasticityLayer(nn.Module):
    """
    Layer with Hebbian-inspired neuroplasticity.

    This layer implements synaptic plasticity based on Hebbian learning
    principles ("neurons that fire together, wire together").
    """

    def __init__(self, hidden_size: int):
        """
        Initialize neuroplasticity layer.

        :param hidden_size: Size of hidden layer
        """
        super().__init__()
        # Plastic connection weights (modified by Hebbian learning)
        self.plastic_weights = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        nn.init.xavier_normal_(self.plastic_weights)

        # Plasticity rate controller
        self.plasticity_rate = nn.Parameter(torch.ones(1) * 0.1)

        # Fixed pathway
        self.fixed_projection = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, x: torch.Tensor, brain_state: Optional[BrainState] = None
    ) -> torch.Tensor:
        """
        Apply neuroplastic processing.

        :param x: Input tensor
        :param brain_state: Optional BrainState for modulation
        :return: Processed tensor
        """
        # Fixed pathway
        fixed_out = self.fixed_projection(x)

        # Plastic pathway
        plastic_out = F.linear(x, self.plastic_weights)

        # Combine pathways
        combined = fixed_out + plastic_out

        # Apply Hebbian update during training
        if self.training:
            # Basic Hebbian rule: weight += learning_rate * (pre * post)
            pre_activation = x.detach()
            post_activation = combined.detach()

            # Outer product for Hebbian weight changes
            batch_size = x.shape[0]
            hebbian_adjustment = torch.zeros_like(self.plastic_weights)

            for i in range(batch_size):
                # Compute pre-post correlations for each item in batch
                pre = pre_activation[i].unsqueeze(1)  # [hidden_size, 1]
                post = post_activation[i].unsqueeze(0)  # [1, hidden_size]
                hebbian_adjustment += torch.bmm(
                    pre.view(-1, pre.size(-1), 1), post.view(-1, 1, post.size(-1))
                ).mean(dim=0)

            hebbian_adjustment /= batch_size

            # Apply adjustment with plasticity rate
            plasticity_factor = self.plasticity_rate

            # Modulate plasticity with brain state if provided
            if brain_state is not None:
                # Use Parameter's data attribute for in-place modification to maintain Parameter type
                plasticity_factor_value = (
                    plasticity_factor * brain_state.hebbian_strength
                )
                # Modify .data directly to avoid type issues
                plasticity_factor.data = plasticity_factor_value.detach()

            with torch.no_grad():
                # Use .data to ensure we're updating the underlying tensor, not the Parameter itself
                adjustment = plasticity_factor * hebbian_adjustment
                # Direct tensor operation on .data
                self.plastic_weights.data.add_(
                    adjustment
                )  # Using add_ in-place operation

                # Stabilize weights through soft normalization
                weight_norm = torch.norm(self.plastic_weights.data)
                if weight_norm > 1.0:
                    # Using div_ in-place operation
                    self.plastic_weights.data.div_(weight_norm)

        # Explicitly return Tensor
        output_tensor: torch.Tensor = combined
        return output_tensor

    def get_plasticity_rate(self) -> torch.Tensor:
        """
        Get current plasticity rate.

        :return: Plasticity rate tensor
        """
        return self.plasticity_rate


class OpenGrammarModel(nn.Module):
    """
    Brain-inspired neural network for text generation.

    This model implements a neuromorphic architecture with
    energy constraints and neuroplasticity.
    """

    def __init__(
        self, hidden_size: int = 256, layer_count: int = 4, byte_embedding: bool = True
    ):
        """
        Initialize OpenGrammar model.

        :param hidden_size: Hidden layer size
        :param layer_count: Number of neural layers
        :param byte_embedding: Whether to use byte-level embedding
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_count = layer_count
        self.use_byte_embedding = byte_embedding

        # Initialize attributes to satisfy type checker
        self.embedding: Optional[nn.Embedding] = None
        self.embedding_projection: Optional[nn.Linear] = None

        # Set appropriate embedding method based on configuration
        if self.use_byte_embedding:
            self.embedding = nn.Embedding(256, hidden_size)
        else:
            self.embedding_projection = nn.Linear(256, hidden_size)

        # Neuromodulatory system
        self.neuromodulation = NeurotransmitterModule(hidden_size)

        # Layer components
        self.layers = nn.ModuleList()
        for _ in range(layer_count):
            layer_group = nn.ModuleDict(
                {
                    "attention": GainControlAttention(hidden_size=hidden_size),
                    "energy": EnergyConstrainedLayer(
                        hidden_size=hidden_size, input_size=hidden_size
                    ),
                    "plasticity": NeuroplasticityLayer(hidden_size=hidden_size),
                    "norm": nn.LayerNorm(hidden_size),
                }
            )
            self.layers.append(layer_group)

        # Output projection for byte prediction
        self.output_projection = nn.Linear(hidden_size, 256)

    def forward(
        self, batch: Dict[str, torch.Tensor], brain_state: Optional[BrainState] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Shape glossary:
        - B: Batch size
        - S: Sequence length
        - H: Hidden dimension size
        - V: Vocabulary size (256 for byte-level)

        :param batch: Batch dictionary with 'text' key
        :param brain_state: Optional BrainState for neuromodulation
        :return: Output logits [B, S, V]
        """
        x = batch["text"]  # [B, S]

        # Embedding
        if self.embedding is not None:
            x = x.long()  # Ensure x is of type torch.LongTensor for embedding
            x = self.embedding(x)  # [B, S, H]
        else:
            # Convert bytes to one-hot and project
            x_one_hot = F.one_hot(x, num_classes=256).float()  # [B, S, 256]
            if self.embedding_projection is not None:
                x = self.embedding_projection(x_one_hot)  # [B, S, H]
            else:
                raise ValueError("Both embedding and embedding_projection are None")

        # Apply neuromodulation
        x, _ = self.neuromodulation(x, brain_state)

        # Process through layers
        for layer in self.layers:
            # Energy-constrained projection
            energy_out = layer["energy"](x)

            # Gain control attention
            attention_out = layer["attention"](energy_out, brain_state)

            # Apply neuroplasticity layer
            plastic_out = layer["plasticity"](x, brain_state)

            # Add residual and normalize
            x = layer["norm"](x + plastic_out)

        # Project to output vocabulary
        logits: torch.Tensor = self.output_projection(x)  # [B, S, V]

        return logits

    def energy_profile(self) -> torch.Tensor:
        """
        Get energy usage profile for all layers.

        :return: Average energy usage tensor
        """
        energy_values = torch.stack(
            [layer["energy"].get_energy_usage() for layer in self.layers]
        )
        return energy_values.mean()

    def plasticity_profile(self) -> torch.Tensor:
        """
        Get plasticity rate profile for all layers.

        :return: Average plasticity rate tensor
        """
        plasticity_values = torch.stack(
            [layer["plasticity"].get_plasticity_rate() for layer in self.layers]
        )
        return plasticity_values.mean()

    def update_brain_state(
        self, brain_state: BrainState, stimulus: str, response: Optional[str] = None
    ) -> BrainState:
        """
        Update brain state based on stimulus-response pair.

        :param brain_state: BrainState object to update
        :param stimulus: Input stimulus (text)
        :param response: Optional response (text)
        :return: Updated BrainState
        """
        # Update brain state
        brain_state.update(stimulus, response)

        # Apply brain state to plastic parameters
        brain_state.apply_to_parameters(self.named_parameters())

        return brain_state


if __name__ == "__main__":
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from opengrammar.brain import BrainState

    # Initialize Rich console
    console = Console()

    console.print(
        Panel.fit(
            "[bold green]OpenGrammar Model Check[/bold green]", border_style="green"
        )
    )

    try:
        # Initialize model with smaller dimensions for quick testing
        console.print("[yellow]Creating OpenGrammarModel...[/yellow]")
        model = OpenGrammarModel(hidden_size=64, layer_count=2)

        # Display model information
        model_info = Table(title="Model Configuration")
        model_info.add_column("Parameter", style="cyan")
        model_info.add_column("Value", style="magenta")

        model_info.add_row("Hidden Size", str(model.hidden_size))
        model_info.add_row("Layer Count", str(model.layer_count))
        model_info.add_row("Byte Embedding", str(model.use_byte_embedding))
        model_info.add_row(
            "Parameter Count", f"{sum(p.numel() for p in model.parameters()):,}"
        )

        console.print(model_info)

        # Create a sample batch
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("[yellow]Creating sample batch...", total=None)

            # Create a small batch for testing
            batch_size = 2
            seq_length = 16

            # Create random byte sequences
            sample_text = torch.randint(
                0, 256, (batch_size, seq_length), dtype=torch.int64
            )

            # Create batch dictionary
            batch = {"text": sample_text}

        # Display sample batch
        batch_info = Table(title="Sample Batch")
        batch_info.add_column("Key", style="cyan")
        batch_info.add_column("Shape", style="magenta")
        batch_info.add_column("Data Type", style="green")
        batch_info.add_column("Sample (first sequence)", style="blue")

        for key, tensor in batch.items():
            # Convert first sequence bytes to ASCII where possible, for better visualization
            sample_str = ""
            for b in tensor[0][:10].tolist():  # First 10 bytes of first sequence
                if 32 <= b <= 126:  # Printable ASCII
                    sample_str += chr(b)
                else:
                    sample_str += f"\\x{b:02x}"  # Hex for non-printable

            if len(tensor[0]) > 10:
                sample_str += "..."

            batch_info.add_row(key, str(tensor.shape), str(tensor.dtype), sample_str)

        console.print(batch_info)

        # Create a basic brain state
        brain_state = BrainState(arousal=0.7, engagement=0.8, hebbian_strength=0.5)

        brain_info = Table(title="Brain State")
        brain_info.add_column("Parameter", style="cyan")
        brain_info.add_column("Value", style="magenta")

        brain_info.add_row("Arousal", str(brain_state.arousal))
        brain_info.add_row("Engagement", str(brain_state.engagement))
        brain_info.add_row("Hebbian Strength", str(brain_state.hebbian_strength))

        console.print(brain_info)

        # Run forward pass
        console.print("\n[yellow]Running forward pass...[/yellow]")
        with torch.no_grad():
            outputs = model(batch, brain_state)

        # Display outputs
        output_info = Table(title="Model Outputs")
        output_info.add_column("Shape", style="magenta")
        output_info.add_column("Data Type", style="green")
        output_info.add_column("Output Range", style="blue")

        output_info.add_row(
            str(outputs.shape),
            str(outputs.dtype),
            f"Min: {outputs.min().item():.4f}, Max: {outputs.max().item():.4f}",
        )

        console.print(output_info)

        # Display energy and plasticity metrics
        metrics = Table(title="Model Metrics")
        metrics.add_column("Metric", style="cyan")
        metrics.add_column("Value", style="magenta")

        metrics.add_row("Energy Profile", f"{model.energy_profile().item():.6f}")
        metrics.add_row(
            "Plasticity Profile", f"{model.plasticity_profile().item():.6f}"
        )

        console.print(metrics)

        # Test specific components
        console.print("\n[yellow]Testing model components...[/yellow]")

        # Test neuromodulatory system
        if model.embedding is not None:
            neuromod_input = model.embedding(batch["text"].long())
            neuromod_output, modulation = model.neuromodulation(
                neuromod_input, brain_state
            )

            console.print(
                f"[green]✓ Neuromodulation - Dopamine level: {modulation['dopamine'].item():.4f}[/green]"
            )
            console.print(
                f"[green]✓ Neuromodulation - Norepinephrine level: {modulation['norepinephrine'].item():.4f}[/green]"
            )

        # Test attention mechanism
        for i, layer in enumerate(model.layers):
            attention_output = layer["attention"](neuromod_output, brain_state)
            console.print(
                f"[green]✓ Attention layer {i+1} output shape: {attention_output.shape}[/green]"
            )

        # Update brain state with some text
        console.print("\n[yellow]Testing brain state update...[/yellow]")
        updated_state = model.update_brain_state(
            brain_state,
            stimulus="Hello, OpenGrammar!",
            response="This is a test response.",
        )

        # Show brain state change
        updated_info = Table(title="Updated Brain State")
        updated_info.add_column("Parameter", style="cyan")
        updated_info.add_column("Original", style="blue")
        updated_info.add_column("Updated", style="green")

        updated_info.add_row(
            "Arousal", str(brain_state.arousal), str(updated_state.arousal)
        )
        updated_info.add_row(
            "Engagement", str(brain_state.engagement), str(updated_state.engagement)
        )
        updated_info.add_row(
            "Hebbian Strength",
            str(brain_state.hebbian_strength),
            str(updated_state.hebbian_strength),
        )

        console.print(updated_info)

        console.print(
            Panel(
                "[bold green]✓ OpenGrammarModel is working correctly![/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel(f"[bold red]Error: {str(e)}[/bold red]", border_style="red")
        )
        import traceback

        console.print(traceback.format_exc(), style="red")
