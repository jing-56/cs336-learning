import pickle
import regex
from collections import Counter
from pathlib import Path
from collections import defaultdict
# 作业提供的预分词规则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def build_pair_index(words):
    pair_index = defaultdict(set)

    for word_id, tokens in enumerate(words):
        for pair in zip(tokens, tokens[1:]):
            pair_index[pair].add(word_id)

    return pair_index
# 逐个获取片段，并统计出现次数
def merge_pair(byte_counts, pair):
    left, right = pair
    merged_token = left + right

    new_byte_counts = Counter()

    for tokens, count in byte_counts.items():
        new_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] == left
                and tokens[i + 1] == right
            ):
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        new_byte_counts[tuple(new_tokens)] += count

    return new_byte_counts
def pretokenize_counts(text, special_tokens):
    # 按特殊标记划分普通文本
    special_pattern = "|".join(
        regex.escape(token)
        for token in special_tokens
    )

    if special_tokens:
        segments = regex.split(special_pattern, text)
    else:
        segments = [text]

    # 统计普通文本中的预分词片段
    counts = Counter()

    for segment in segments:
        for match in regex.finditer(PAT, segment):
            counts[match.group()] += 1

    return counts


# 调用函数，取得频率表
def train_bpe(input_path, vocab_size, special_tokens):
    # 读取文本，统计预分词片段
    text = Path(input_path).read_text(encoding="utf-8")
    counts = pretokenize_counts(text, special_tokens)

    # 转换为字节 token 序列
    byte_counts = {}
    for piece, count in counts.items():
        tokens = tuple(
            bytes([value])
            for value in piece.encode("utf-8")
        )
        byte_counts[tokens] = count

    # 初始化基础词表和特殊 token
    vocab = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError("目标词表太小，容不下基础和特殊 token")

    merges = []
    # 给每种片段分配固定编号
    words = list(byte_counts)

    # 保存各片段在语料中的出现次数
    word_counts = [byte_counts[tokens] for tokens in words]

    # 建立“token 对 → 片段编号”的目录
    pair_index = build_pair_index(words)

    # 初始时，对所有片段统计一次
    pair_counts = Counter()

    for word_id, tokens in enumerate(words):
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += word_counts[word_id]

    while len(vocab) < vocab_size and pair_counts:
        # 选择本轮要合并的 token 对
        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair)
        )

        left, right = best_pair
        vocab[len(vocab)] = left + right
        merges.append(best_pair)

        # 查目录：只找到包含这一对的片段
        affected_ids = list(pair_index[best_pair])

        for word_id in affected_ids:
            old_tokens = words[word_id]
            weight = word_counts[word_id]

            # 1. 撤掉这个片段原来的计数和目录记录
            old_pairs = Counter(
                zip(old_tokens, old_tokens[1:])
            )

            for pair, occurrences in old_pairs.items():
                pair_counts[pair] -= occurrences * weight

                if pair_counts[pair] == 0:
                    del pair_counts[pair]

                pair_index[pair].discard(word_id)

                if not pair_index[pair]:
                    del pair_index[pair]

            # 2. 只合并当前这个片段
            updated = merge_pair(
                {old_tokens: weight},
                best_pair
            )
            new_tokens = next(iter(updated))
            words[word_id] = new_tokens

            # 3. 加入修改后的计数和目录记录
            new_pairs = Counter(
                zip(new_tokens, new_tokens[1:])
            )

            for pair, occurrences in new_pairs.items():
                pair_counts[pair] += occurrences * weight
                pair_index[pair].add(word_id)

    return vocab, merges


# 直接运行本文件时，执行这个小实验
if __name__ == "__main__":
    input_path = Path(__file__).with_name("corpus_demo.txt")

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=259,
        special_tokens=["<|endoftext|>"]
    )

    print("最终词表大小：", len(vocab))
    print("合并规则：", merges)
    output_path = Path(__file__).with_name("bpe_demo.pkl")

    with output_path.open("wb") as file:
        pickle.dump(
            {"vocab": vocab, "merges": merges},
            file
        )

    print("训练结果已保存到：", output_path)
    vocab_path = Path(__file__).with_name("bpe_vocab.pkl")
    merges_path = Path(__file__).with_name("bpe_merges.pkl")

    with vocab_path.open("wb") as file:
        pickle.dump(vocab, file)

    with merges_path.open("wb") as file:
        pickle.dump(merges, file)