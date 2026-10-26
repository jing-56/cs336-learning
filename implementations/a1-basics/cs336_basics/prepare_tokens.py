from pathlib import Path

import numpy as np

from cs336_basics.tokenizer_demo import Tokenizer


folder = Path(__file__).resolve().parent
project_root = folder.parent

# 加载刚才在 5 MB 样本上训练的分词器
tokenizer = Tokenizer.from_files(
    folder / "tinystories_sample_10k_vocab.pkl",
    folder / "tinystories_sample_10k_merges.pkl",
    special_tokens=["<|endoftext|>"]
)

# 确认所有 ID 都能用 uint16 表示
assert min(tokenizer.vocab) >= 0
assert max(tokenizer.vocab) <= np.iinfo(np.uint16).max

# 先转换较小的故事文件
input_path = (
    project_root / "tests" / "fixtures" / "tinystories_sample.txt"
)

with input_path.open("r", encoding="utf-8") as file:
    tokens = np.fromiter(
        tokenizer.encode_iterable(file),
        dtype=np.uint16
    )

# 保存为 NumPy 文件
output_path = folder / "tinystories_sample_tokens.npy"
np.save(output_path, tokens)

# 小样本检查：转回文本，确认没有损失
original = input_path.read_text(encoding="utf-8")
restored = tokenizer.decode(tokens.tolist())

print("token 数量：", len(tokens))
print("数据类型：", tokens.dtype)
print("数组数据大小：", tokens.nbytes, "字节")
print("完整还原：", restored == original)
print("保存位置：", output_path)