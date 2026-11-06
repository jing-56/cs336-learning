"""Run the CS336 trainer on token arrays that can exceed 32-bit indexing."""

import numpy as np
import torch

from cs336_basics import train_cli
from cs336_basics.data import get_batch as original_get_batch


def get_batch(dataset, batch_size, context_length, device):
    upper = len(dataset) - context_length
    if upper <= np.iinfo(np.int32).max:
        return original_get_batch(dataset, batch_size, context_length, device)
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("Batch size and context length must be positive")
    starts = np.random.randint(0, upper, size=batch_size, dtype=np.int64)
    windows = np.stack([dataset[start:start + context_length + 1] for start in starts])
    ids = torch.tensor(windows, dtype=torch.long, device=device)
    return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


if __name__ == "__main__":
    train_cli.get_batch = get_batch
    train_cli.main()
