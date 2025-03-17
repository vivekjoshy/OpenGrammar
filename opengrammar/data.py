from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class MiniPileDataset(Dataset):
    def __init__(self, root_path: Path, validation: bool = False):
        """
        Initialize the MiniPileDataset with the root path to the MiniPile dataset.

        :param root_path: The root path to the MiniPile dataset.
        """
        self.root_path = root_path
        self.validation = validation

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]: ...


class MiniPileCollate:
    def __init__(self): ...

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]: ...


if __name__ == "__main__":
    ...
