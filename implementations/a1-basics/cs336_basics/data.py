import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device):
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("批次大小和上下文长度必须大于零")

    if len(dataset) <= context_length:
        raise ValueError("数据长度至少需要比上下文长度多一个 token")

    # 上界不包含在采样范围内，确保最后一个目标不会越界
    starts = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size
    )

    x = np.stack([
        dataset[start:start + context_length]
        for start in starts
    ])

    y = np.stack([
        dataset[start + 1:start + context_length + 1]
        for start in starts
    ])

    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    return x, y