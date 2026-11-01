"""用新建的 FP32 模型训练一步，统计参数、梯度和 AdamW 状态张量。"""

import torch

from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW
from cs336_basics.train_rope_ablation import MODEL_CONFIG
from cs336_basics.transformer_lm import TransformerLM


MIB = 1024 ** 2


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def print_storage(label, model, optimizer):
    parameters = sum(tensor_bytes(p) for p in model.parameters())
    gradients = sum(tensor_bytes(p.grad) for p in model.parameters() if p.grad is not None)
    first_moment = sum(tensor_bytes(state["m"]) for state in optimizer.state.values() if "m" in state)
    second_moment = sum(tensor_bytes(state["v"]) for state in optimizer.state.values() if "v" in state)
    total = parameters + gradients + first_moment + second_moment
    print(f"{label}：参数 {parameters / MIB:.2f}、梯度 {gradients / MIB:.2f}、"
          f"m {first_moment / MIB:.2f}、v {second_moment / MIB:.2f}、"
          f"小计 {total / MIB:.2f} MiB")
    return parameters, gradients, first_moment, second_moment


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = TransformerLM(**MODEL_CONFIG, dtype=torch.float32)
    optimizer = AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    p_count = sum(p.numel() for p in model.parameters())
    expected_bytes = p_count * 4
    print("参数量：", f"{p_count:,}")
    print("每个 float32 占 4 字节；1 MiB = 1024×1024 字节。")
    print("以下仅统计四类张量的元素存储，不是进程总内存或峰值内存。\n")
    before = print_storage("刚创建", model, optimizer)
    if before != (expected_bytes, 0, 0, 0):
        raise RuntimeError("初始化张量占用与预期不符")

    # 随机 token 仅用于演示内存；目标仍是输入右移一位，不用来评估模型效果。
    b, t = 4, 64
    token_stream = torch.randint(0, MODEL_CONFIG["vocab_size"], (b, t + 1))
    x, y = token_stream[:, :-1], token_stream[:, 1:]
    logits = model(x)
    loss = cross_entropy(logits, y)
    if not torch.isfinite(loss).item():
        raise RuntimeError("演示 loss 非有限")
    loss.backward()
    after_backward = print_storage("反向传播后", model, optimizer)
    if after_backward != (expected_bytes, expected_bytes, 0, 0):
        raise RuntimeError("梯度张量占用与预期不符")

    gradient_clipping(model.parameters(), max_l2_norm=1.0)
    optimizer.step()
    after_step = print_storage("第一次更新后", model, optimizer)
    if after_step != (expected_bytes,) * 4:
        raise RuntimeError("AdamW 状态张量占用与预期不符")
    optimizer.zero_grad(set_to_none=True)
    after_clear = print_storage("清除梯度后", model, optimizer)
    if after_clear != (expected_bytes, 0, expected_bytes, expected_bytes):
        raise RuntimeError("清梯度后张量占用与预期不符")

    print(f"\n单份参数：{expected_bytes:,} 字节，约 {expected_bytes / MIB:.2f} MiB")
    print(f"参数+梯度+m+v：{expected_bytes * 4:,} 字节，约 {expected_bytes * 4 / MIB:.2f} MiB")
    print(f"单个 logits 张量：形状 {tuple(logits.shape)}，约 {tensor_bytes(logits) / MIB:.2f} MiB")
    print("上面的 logits 是实际张量的大小，不是所有中间激活的合计。")
    print("\n输入规模变化的形状估算（没有运行更长上下文）：")
    for batch_size, seq_len in ((4, 64), (8, 64), (4, 128)):
        logits_size = batch_size * seq_len * MODEL_CONFIG["vocab_size"] * 4
        attention_size = batch_size * MODEL_CONFIG["num_heads"] * seq_len ** 2 * 4
        print(f"B={batch_size}，T={seq_len}：单个 logits {logits_size / MIB:.2f} MiB，"
              f"单层单个注意力分数或权重张量 {attention_size / MIB:.2f} MiB")
    print("\n实际训练还包含其他中间激活、临时张量、固定 buffer 和运行库等开销。")
    print("清除梯度后，grad 字段不再持有张量；不表示系统内存立即下降。")
    print("演示只更新新建模型一步，不加载或覆盖既有训练存档。")


if __name__ == "__main__":
    main()
