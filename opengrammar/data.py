import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


class MiniPileDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        root_path: Path,
        validation: bool = False,
        sequence_length: int = 1024,
        random_seed: int = 7,
    ) -> None:
        """
        Initialize the MiniPileDataset with the root path to the MiniPile dataset.

        The MiniPile dataset is expected to be in parquet format with files following the pattern:
        - train-XXXXX-of-XXXXX-*.parquet for the training set
        - validation-XXXXX-of-XXXXX-*.parquet for the validation set

        :param root_path: The root path to the MiniPile dataset
        :param validation: Whether to use the validation split
        :param sequence_length: The sequence length to use for each sample
        :param random_seed: Random seed for reproducibility
        """
        self.root_path = root_path
        self.validation = validation
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        self.cache: Dict[Path, pd.DataFrame] = {}

        # Determine which files to use based on validation flag
        if validation:
            file_pattern = "validation-*-of-*-*.parquet"
        else:
            file_pattern = "train-*-of-*-*.parquet"

        # Find all matching parquet files
        self.file_paths = list(self.root_path.glob(file_pattern))

        if len(self.file_paths) == 0:
            raise ValueError(f"No {file_pattern} files found in {self.root_path}")

        # Sort files for deterministic behavior
        self.file_paths = sorted(self.file_paths)

        # Count total samples
        self.total_samples = 0
        for file_path in self.file_paths:
            # Load the parquet file
            df = self._get_dataframe(file_path)

            # Count number of samples
            file_samples = len(df)
            self.total_samples += file_samples

        # Set random seed
        random.seed(random_seed)

    def _get_dataframe(self, file_path: Path) -> pd.DataFrame:
        """
        Get the dataframe for a file, using cache if available.

        :param file_path: Path to the parquet file
        :return: Pandas DataFrame
        """
        # Use cached dataframe if available
        if file_path in self.cache:
            return self.cache[file_path]

        # Load the parquet file
        df = pd.read_parquet(file_path)

        # Cache the dataframe
        self.cache[file_path] = df

        return df

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        :return: Number of samples
        """
        return self.total_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Shape glossary:
        - S: Sequence length

        :param index: Index of the sample
        :return: Dictionary with 'text' and 'target' tensors of shape [S]
        """
        # Find which file contains this index
        for file_path in self.file_paths:
            df = self._get_dataframe(file_path)
            file_samples = len(df)

            if index < file_samples:
                # This file contains our sample
                break

            index -= file_samples
        else:
            # This should never happen if __len__ is implemented correctly
            raise IndexError(f"Index {index} out of range")

        # Get the text from the dataframe
        text = df.iloc[index]["text"]

        # Handle shorter sequences by repeating
        if len(text) < self.sequence_length + 1:
            text = text * ((self.sequence_length + 1) // len(text) + 1)

        # Choose a random starting point
        start_idx = random.randint(0, len(text) - self.sequence_length - 1)

        # Extract sequence and target
        source_seq = text[start_idx : start_idx + self.sequence_length]
        target_seq = text[start_idx + 1 : start_idx + self.sequence_length + 1]

        # Convert to tensors using int64 instead of uint8 to handle all Unicode characters
        source_bytes = torch.tensor([ord(c) for c in source_seq], dtype=torch.int64)
        target_bytes = torch.tensor([ord(c) for c in target_seq], dtype=torch.int64)

        return {
            "text": source_bytes,  # [S]
            "target": target_bytes,  # [S]
        }


class MiniPileCollate:
    """
    Collate function for the MiniPile dataset.

    This handles batching of samples from the dataset.
    """

    def __init__(self, sequence_length: int = 1024) -> None:
        """
        Initialize the collate function.

        :param sequence_length: The sequence length for each sample
        """
        self.sequence_length = sequence_length

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Shape glossary:
        - B: Batch size
        - S: Sequence length

        :param batch: List of samples from the dataset
        :return: Batched dictionary with 'text' and 'target' tensors of shape [B, S]
        """
        # Stack tensors along batch dimension
        text_batch = torch.stack([sample["text"] for sample in batch])
        target_batch = torch.stack([sample["target"] for sample in batch])

        return {
            "text": text_batch,  # [B, S]
            "target": target_batch,  # [B, S]
        }


class TokenSeqDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    Dataset for token sequences.

    This dataset handles sequences of token IDs for training
    the model without using a tokenizer.
    """

    def __init__(
        self,
        sequences: List[List[int]],
        sequence_length: int = 1024,
        vocab_size: int = 256,
    ) -> None:
        """
        Initialize the token sequence dataset.

        :param sequences: List of token sequences (lists of integers)
        :param sequence_length: Maximum sequence length
        :param vocab_size: Size of the vocabulary
        """
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        :return: Number of samples
        """
        return len(self.sequences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Shape glossary:
        - S: Sequence length

        :param index: Index of the sample
        :return: Dictionary with 'text' and 'target' tensors of shape [S]
        """
        # Get sequence and ensure it's long enough
        sequence = self.sequences[index]
        if len(sequence) < self.sequence_length + 1:
            # Repeat sequence if it's too short
            sequence = sequence * ((self.sequence_length + 1) // len(sequence) + 1)

        # Get input and target sequences
        input_seq = sequence[: self.sequence_length]
        target_seq = sequence[1 : self.sequence_length + 1]

        # Convert to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.int64)
        target_tensor = torch.tensor(target_seq, dtype=torch.int64)

        return {
            "text": input_tensor,  # [S]
            "target": target_tensor,  # [S]
        }


if __name__ == "__main__":
    # Import Rich for better visualization
    from pathlib import Path

    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import track
    from rich.table import Table

    # Initialize Rich console
    console = Console()

    # Root Path - use the actual path from your error message
    root_path = Path("resources/minipile")

    console.print(
        Panel.fit(
            "[bold green]MiniPile Dataset Check[/bold green]", border_style="green"
        )
    )

    try:
        # Create dataset instance
        console.print("[yellow]Creating dataset...[/yellow]")
        dataset = MiniPileDataset(
            root_path=root_path, sequence_length=128
        )  # Shorter sequence for demo

        console.print(
            f"[green]Dataset created successfully with [bold]{len(dataset)}[/bold] samples![/green]"
        )

        # Create a table to show sample information
        table = Table(title="Dataset Sample Preview")
        table.add_column("Sample Index", style="cyan")
        table.add_column("Text Shape", style="magenta")
        table.add_column("Target Shape", style="magenta")
        table.add_column("Text Preview", style="green")
        table.add_column("Data Type", style="yellow")

        # Show a few samples
        num_samples_to_show = min(5, len(dataset))

        for i in track(range(num_samples_to_show), description="Fetching samples"):
            sample = dataset[i]
            text_tensor = sample["text"]
            target_tensor = sample["target"]

            # Convert first few bytes back to characters for preview
            preview_length = min(20, len(text_tensor))
            text_preview = "".join(
                chr(b) for b in text_tensor[:preview_length].tolist()
            )
            if preview_length < len(text_tensor):
                text_preview += "..."

            table.add_row(
                str(i),
                str(text_tensor.shape),
                str(target_tensor.shape),
                text_preview,
                str(text_tensor.dtype),
            )

        console.print(table)

        # Show character code ranges (helpful for debugging)
        code_info = Table(title="Character Code Information")
        code_info.add_column("Sample", style="cyan")
        code_info.add_column("Min Value", style="magenta")
        code_info.add_column("Max Value", style="magenta")
        code_info.add_column("Mean Value", style="green")

        sample = dataset[0]
        text_tensor = sample["text"]

        code_info.add_row(
            "0",
            str(text_tensor.min().item()),
            str(text_tensor.max().item()),
            str(text_tensor.float().mean().item()),
        )

        console.print(code_info)

        # Test the collate function
        console.print("\n[yellow]Testing batch collation...[/yellow]")
        collate_fn = MiniPileCollate(sequence_length=128)
        batch_size = 3

        # Manually create a small batch
        batch_indices = list(range(min(batch_size, len(dataset))))
        batch_samples = [dataset[i] for i in batch_indices]

        # Collate the batch
        batch = collate_fn(batch_samples)

        # Show batch info
        batch_table = Table(title="Batch Information")
        batch_table.add_column("Key", style="cyan")
        batch_table.add_column("Shape", style="magenta")
        batch_table.add_column("Dtype", style="green")

        for key, tensor in batch.items():
            batch_table.add_row(key, str(tensor.shape), str(tensor.dtype))

        console.print(batch_table)
        console.print(
            Panel(
                "[bold green]✓ Dataset and batch collation working correctly![/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel(f"[bold red]Error: {str(e)}[/bold red]", border_style="red")
        )
        import traceback

        console.print(traceback.format_exc(), style="red")
