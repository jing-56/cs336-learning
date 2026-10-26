from pathlib import Path
from time import perf_counter
import pickle
from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer_demo import Tokenizer
import psutil

project_root = Path(__file__).resolve().parents[1]
input_path = (
    project_root
    / "tests"
    / "fixtures"
    / "tinystories_sample_5M.txt"
)

special_tokens = ["<|endoftext|>"]
process = psutil.Process()

# 训练开始前，程序当前占用的物理内存
baseline_memory = process.memory_info().rss
print("开始训练，语料约 5 MB……", flush=True)
start = perf_counter()

vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=10000,
    special_tokens=special_tokens
)

training_seconds = perf_counter() - start
# 训练结束后立即读取，尚未进行后面的评估
memory = process.memory_info()
mib = 1024 ** 2

print(f"训练前内存：{baseline_memory / mib:.2f} MiB")
print(f"训练后当前内存：{memory.rss / mib:.2f} MiB")
print(f"截至训练结束的进程峰值内存：{memory.peak_wset / mib:.2f} MiB")
tokenizer = Tokenizer(vocab, merges, special_tokens)

evaluation_text = (
    "A robot opened the window and watched the stars. "
    "It wanted to learn why the sky was dark at night."
)

token_ids = tokenizer.encode(evaluation_text)
byte_count = len(evaluation_text.encode("utf-8"))

print("实际词表大小：", len(vocab))
print("合并规则数量：", len(merges))
print(f"训练耗时：{training_seconds:.3f} 秒")
print("评估 token 数：", len(token_ids))
print(f"bytes/token：{byte_count / len(token_ids):.2f}")
print("完整还原：", tokenizer.decode(token_ids) == evaluation_text)
# 查看最长的普通 token，排除特殊标记
special_bytes = {
    token.encode("utf-8")
    for token in special_tokens
}

longest_token = max(
    (
        token
        for token in vocab.values()
        if token not in special_bytes
    ),
    key=len
)

print("最长普通 token：", longest_token)
print("它的字节长度：", len(longest_token))

# 为这次样本实验单独保存文件
vocab_path = Path(__file__).with_name(
    "tinystories_sample_10k_vocab.pkl"
)
merges_path = Path(__file__).with_name(
    "tinystories_sample_10k_merges.pkl"
)

with vocab_path.open("wb") as file:
    pickle.dump(vocab, file)

with merges_path.open("wb") as file:
    pickle.dump(merges, file)

print("词表已保存：", vocab_path)
print("合并规则已保存：", merges_path)