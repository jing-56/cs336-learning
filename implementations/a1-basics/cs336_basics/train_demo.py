from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.checkpoint import save_checkpoint


def main():
    # 固定随机种子，方便重复观察
    np.random.seed(42)
    torch.manual_seed(42)

    # 这个小实验使用 CPU
    torch.set_num_threads(4)
    device = "cpu"

    # 读取之前准备好的 token 数据
    data_path = Path(__file__).with_name(
        "tinystories_sample_tokens.npy"
    )
    data = np.load(data_path, mmap_mode="r")

    # 创建一个较小的模型
    model = TransformerLM(
        vocab_size=10000,
        context_length=16,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        device=device,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=0.01,
    )

    # 只抽取一次：后面反复学习这一批数据
    x, y = get_batch(
        dataset=data,
        batch_size=2,
        context_length=16,
        device=device,
    )

    model.train()

    with torch.no_grad():
        initial_loss = cross_entropy(model(x), y).item()

    print("数据 token 数：", len(data))
    print("固定批次形状：", x.shape)
    print(f"训练前 loss：{initial_loss:.4f}")

    num_steps = 100

    for step in range(1, num_steps + 1):
        # 1. 清空旧梯度
        optimizer.zero_grad()

        # 2. 预测并计算损失
        logits = model(x)
        loss = cross_entropy(logits, y)

        # 3. 计算梯度
        loss.backward()

        # 4. 裁剪梯度
        gradient_clipping(
            model.parameters(),
            max_l2_norm=1.0,
        )

        # 5. 更新参数
        optimizer.step()

        if step == 1 or step % 20 == 0:
            print(f"step={step:3d}，loss={loss.item():.4f}")

    # 用更新后的模型重新检查同一批数据
    with torch.no_grad():
        logits = model(x)
        final_loss = cross_entropy(logits, y).item()

        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == y).float().mean().item()

    print(f"训练后 loss：{final_loss:.4f}")
    print(f"固定批次预测准确率：{accuracy:.2%}")

    # 保存训练状态
    checkpoint_path = Path(__file__).with_name(
        "overfit_demo_checkpoint.pt"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=num_steps,
        out=checkpoint_path,
    )

    print("训练存档已保存：", checkpoint_path)


if __name__ == "__main__":
    main()