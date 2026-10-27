from pathlib import Path
from time import perf_counter

import numpy as np

from cs336_basics.tokenizer_demo import Tokenizer


def encode_and_save(text, tokenizer, output_path, name):
    print(f"开始编码{name}……", flush=True)
    start = perf_counter()

    # 文本 → token ID → NumPy 数组
    token_ids = tokenizer.encode(text)
    tokens = np.asarray(token_ids, dtype=np.uint16)

    # 保存前检查，确保可以完整还原
    restored = tokenizer.decode(tokens.tolist())

    if restored != text:
        raise ValueError(f"{name}无法完整还原，请检查分词器")

    np.save(output_path, tokens)

    elapsed = perf_counter() - start

    print(f"{name} token 数：{len(tokens):,}")
    print(f"{name}数组大小：{tokens.nbytes / 1024**2:.2f} MiB")
    print(f"编码及检查耗时：{elapsed:.2f} 秒")
    print("完整还原：True")
    print("保存位置：", output_path)
    print()


def main():
    folder = Path(__file__).resolve().parent
    project_root = folder.parent

    # 继续使用原来的 10k 分词器
    tokenizer = Tokenizer.from_files(
        folder / "tinystories_sample_10k_vocab.pkl",
        folder / "tinystories_sample_10k_merges.pkl",
        special_tokens=["<|endoftext|>"],
    )

    # 确保 token ID 可以安全存入 uint16
    if not (
        0 <= min(tokenizer.vocab)
        and max(tokenizer.vocab) <= np.iinfo(np.uint16).max
    ):
        raise ValueError("词表中的 ID 超出了 uint16 的范围")

    input_path = (
        project_root
        / "tests"
        / "fixtures"
        / "tinystories_sample_5M.txt"
    )

    text = input_path.read_text(encoding="utf-8")

    # 在接近 90% 的位置，寻找下一个故事结束标记
    eos_text = "<|endoftext|>"
    target_position = int(len(text) * 0.9)
    boundary = text.find(eos_text, target_position)

    if boundary == -1:
        raise ValueError("没有找到合适的故事划分边界")

    split_position = boundary + len(eos_text)

    train_text = text[:split_position]
    val_text = text[split_position:]

    if not train_text or not val_text:
        raise ValueError("训练文本和验证文本都不能为空")

    print("训练集结束标记数：", train_text.count(eos_text))
    print("验证集结束标记数：", val_text.count(eos_text))

    encode_and_save(
        text=train_text,
        tokenizer=tokenizer,
        output_path=folder / "tinystories_5m_train.npy",
        name="训练集",
    )

    encode_and_save(
        text=val_text,
        tokenizer=tokenizer,
        output_path=folder / "tinystories_5m_val.npy",
        name="验证集",
    )


if __name__ == "__main__":
    main()