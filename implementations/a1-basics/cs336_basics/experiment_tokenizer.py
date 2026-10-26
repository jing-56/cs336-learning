from pathlib import Path
from time import perf_counter
from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer_demo import Tokenizer


project_root = Path(__file__).resolve().parents[1]
input_path = (
    project_root / "tests" / "fixtures" / "tinystories_sample.txt"
)

# 这段文本只用于评估，不传给训练函数
evaluation_text = (
    "A robot opened the window and watched the stars. "
    "It wanted to learn why the sky was dark at night."
)

byte_count = len(evaluation_text.encode("utf-8"))
print("评估文本字节数：", byte_count)

for target_size in [300, 500, 1000]:
    # 每次仍然使用同一份故事语料训练
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=target_size,
        special_tokens=["<|endoftext|>"]
    )

    tokenizer = Tokenizer(
        vocab,
        merges,
        special_tokens=["<|endoftext|>"]
    )

    # 用新文本评估
    token_ids = tokenizer.encode(evaluation_text)
    restored = tokenizer.decode(token_ids)

    print(
        f"目标词表={target_size}，"
        f"实际词表={len(vocab)}，"
        f"token数={len(token_ids)}，"
        f"bytes/token={byte_count / len(token_ids):.2f}，"
        f"完整还原={restored == evaluation_text}"
    )
    # 构造较长的测试文本
    benchmark_text = (evaluation_text + "\n") * 1000
    benchmark_bytes = len(benchmark_text.encode("utf-8"))

    # 连续编码三次，测量总耗时
    repeats = 3
    start = perf_counter()

    for _ in range(repeats):
        benchmark_ids = tokenizer.encode(benchmark_text)

    elapsed = perf_counter() - start

    # 总处理量 ÷ 总耗时
    bytes_per_second = benchmark_bytes * repeats / elapsed
    tokens_per_second = len(benchmark_ids) * repeats / elapsed

    print(f"编码吞吐量：{bytes_per_second / 1_000_000:.2f} MB/s")
    print(f"token 吞吐量：{tokens_per_second:,.0f} tokens/s")
    print()