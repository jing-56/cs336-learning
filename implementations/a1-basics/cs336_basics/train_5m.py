from pathlib import Path
import torch
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
import numpy as np
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.data import get_batch
def evaluate_loss(model, batches):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            loss = cross_entropy(logits, y).item()

            total_loss += loss

    model.train()

    return total_loss / len(batches)

def main():
    np.random.seed(42)

    folder = Path(__file__).resolve().parent

    # 数据已经划分好，分别读取即可
    train_data = np.load(
        folder / "tinystories_5m_train.npy",
        mmap_mode="r",
    )

    val_data = np.load(
        folder / "tinystories_5m_val.npy",
        mmap_mode="r",
    )

    print("训练集 token 数：", len(train_data))
    print("验证集 token 数：", len(val_data))

    # 分别从两份数据中抽取批次
    train_x, train_y = get_batch(
        dataset=train_data,
        batch_size=4,
        context_length=64,
        device="cpu",
    )

    val_x, val_y = get_batch(
        dataset=val_data,
        batch_size=4,
        context_length=64,
        device="cpu",
    )
    torch.manual_seed(42)
    torch.set_num_threads(4)

    model = TransformerLM(
        vocab_size=10000,
        context_length=64,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        device="cpu",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=0.01,
    )

    # 提前抽取 10 个验证批次，之后每次检查都使用它们
    val_batches = [
        get_batch(
            dataset=val_data,
            batch_size=4,
            context_length=64,
            device="cpu",
        )
        for _ in range(10)
    ]

    # 整个训练计划共 1000 步
    num_steps = 1000

    # 本次运行最多训练 900 步，且不超过整个计划的总步数
    steps_this_run = 900

    checkpoint_path = Path(__file__).with_name(
        "train_5m_checkpoint.pt"
    )

    start_step = 0

    # 如果已有存档，就恢复模型、优化器和已完成步数
    if checkpoint_path.exists():
        start_step = load_checkpoint(
            src=checkpoint_path,
            model=model,
            optimizer=optimizer,
        )

        print(f"已恢复存档：已完成 {start_step} 步")
    else:
        print("没有存档，从头训练")

    if start_step >= num_steps:
        print("已经完成全部训练步数")
        return

    current_val_loss = evaluate_loss(model, val_batches)
    print(f"本次训练开始前，平均验证 loss={current_val_loss:.4f}")

    end_step = min(start_step + steps_this_run, num_steps)

    model.train()

    for step in range(start_step + 1, end_step + 1):
        # 每一步重新抽取一个训练批次
        x, y = get_batch(
            dataset=train_data,
            batch_size=4,
            context_length=64,
            device="cpu",
        )
        # 根据当前训练步数，计算本步的学习率
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=3e-3,
            min_learning_rate=3e-4,
            warmup_iters=50,
            cosine_cycle_iters=num_steps,
        )

        # 把计算结果交给优化器
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()

        logits = model(x)
        loss = cross_entropy(logits, y)

        loss.backward()

        gradient_clipping(
            model.parameters(),
            max_l2_norm=1.0,
        )

        optimizer.step()

        if (
                step == start_step + 1
                or step % 20 == 0
                or step == end_step
        ):
            val_loss = evaluate_loss(model, val_batches)

            print(
                f"step={step:3d}，"
                f"lr={lr:.6f}，"
                f"训练 loss={loss.item():.4f}，"
                f"验证 loss={val_loss:.4f}"
            )
    # 当前这一轮训练结束后，保存已经完成的状态
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=end_step,
        out=checkpoint_path,
    )

    print(f"已保存存档：完成第 {end_step} 步")
    print("存档位置：", checkpoint_path)
    print("训练批次形状：", train_x.shape, train_y.shape)
    print("验证批次形状：", val_x.shape, val_y.shape)


if __name__ == "__main__":
    main()