from pathlib import Path

import torch

from cs336_basics.tokenizer_demo import Tokenizer
from cs336_basics.transformer_lm import TransformerLM


def main():
    torch.set_num_threads(4)

    folder = Path(__file__).resolve().parent

    # 1. 加载训练数据使用的同一个分词器
    tokenizer = Tokenizer.from_files(
        folder / "tinystories_sample_10k_vocab.pkl",
        folder / "tinystories_sample_10k_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    # 2. 创建与训练时结构相同的模型
    model = TransformerLM(
        vocab_size=10000,
        context_length=16,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        device="cpu",
    )

    # 3. 加载已经训练好的模型参数
    checkpoint = torch.load(
        folder / "train_loop_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("加载的训练步数：", checkpoint["iteration"])

    # 4. 把提示文本编码为 token ID
    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt)

    token_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
    )

    eos_id = tokenizer.encode("<|endoftext|>")[0]
    max_new_tokens = 40
    temperature =0.8
    top_p = 0.9
    # 5. 每次预测一个 token，再接到已有序列末尾
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 模型一次最多接收 16 个 token
            context = token_ids[:, -model.context_length:]

            logits = model(context)

            # 只使用最后一个位置的预测分数
            next_logits = logits[:, -1, :]

            # 把分数转换成概率
            probabilities = torch.softmax(
                next_logits / temperature,
                dim=-1,
            )

            # 1. 按概率从高到低排序，同时保留原始 token ID
            sorted_probs, sorted_ids = torch.sort(
                probabilities,
                dim=-1,
                descending=True,
            )

            # 2. 计算累计概率
            cumulative_probs = sorted_probs.cumsum(dim=-1)

            if top_p < 1.0:
                # 加入当前候选之前，累计概率还没达到阈值，
                # 就保留当前候选
                previous_probs = cumulative_probs - sorted_probs
                keep = previous_probs < top_p

                # 不保留的候选，概率设为 0
                sorted_probs = sorted_probs.masked_fill(
                    ~keep,
                    0.0,
                )

            # 3. 对保留的概率重新归一化
            sorted_probs = sorted_probs / sorted_probs.sum(
                dim=-1,
                keepdim=True,
            )

            # 4. 抽取候选在排序列表中的位置
            sampled_position = torch.multinomial(
                sorted_probs,
                num_samples=1,
            )

            # 5. 根据排序位置，取回真正的 token ID
            next_id = sorted_ids.gather(
                dim=-1,
                index=sampled_position,
            )

            if next_id.item() == eos_id:
                break

            token_ids = torch.cat(
                [token_ids, next_id],
                dim=1,
            )

    # 6. 将完整序列解码为文字
    result = tokenizer.decode(token_ids[0].tolist())

    print("提示文本：", prompt)
    print("提示 token ID：", prompt_ids)
    print("生成结果：", result)


if __name__ == "__main__":
    main()