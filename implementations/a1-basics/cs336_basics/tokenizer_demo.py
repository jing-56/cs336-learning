import pickle
from pathlib import Path
import regex
from cs336_basics.bpe import PAT, merge_pair

def decode(token_ids, vocab):
    # 根据 ID 取出字节串，再按顺序拼接
    combined = b"".join(vocab[token_id] for token_id in token_ids)

    # 将完整字节串解码为文本
    return combined.decode("utf-8", errors="replace")

def encode_normal(text, vocab, merges):
    # 反向词表：字节内容 → ID
    token_to_id = {
        token: token_id
        for token_id, token in vocab.items()
    }

    # 合并优先级：越早学到，编号越小
    merge_ranks = {
        pair: rank
        for rank, pair in enumerate(merges)
    }

    token_ids = []

    for match in regex.finditer(PAT, text):
        piece = match.group()

        tokens = tuple(
            bytes([value])
            for value in piece.encode("utf-8")
        )

        while len(tokens) > 1:
            # 找出当前相邻对中，有已知合并规则的那些对
            candidates = [
                pair
                for pair in zip(tokens, tokens[1:])
                if pair in merge_ranks
            ]

            if not candidates:
                break

            # 优先执行最早学到的规则
            best_pair = min(
                candidates,
                key=lambda pair: merge_ranks[pair]
            )

            updated = merge_pair({tokens: 1}, best_pair)
            tokens = next(iter(updated))

        # 将最终 token 转成 ID，按顺序追加
        token_ids.extend(
            token_to_id[token]
            for token in tokens
        )

    return token_ids
def encode(text, vocab, merges, special_tokens=None):
    # 没有特殊标记时，直接处理普通文本
    if not special_tokens:
        return encode_normal(text, vocab, merges)

    token_to_id = {
        token: token_id
        for token_id, token in vocab.items()
    }

    # 特殊标记的文本 → 它在词表中的 ID
    special_to_id = {
        token: token_to_id[token.encode("utf-8")]
        for token in special_tokens
    }

    # 较长的标记优先匹配，避免相互包含时先匹配短的
    ordered_specials = sorted(
        special_tokens, key=len, reverse=True
    )

    # 外层括号让 split 保留匹配到的特殊标记
    pattern = "(" + "|".join(
        regex.escape(token)
        for token in ordered_specials
    ) + ")"

    parts = regex.split(pattern, text)

    token_ids = []

    for part in parts:
        if part in special_to_id:
            token_ids.append(special_to_id[part])
        elif part:
            token_ids.extend(
                encode_normal(part, vocab, merges)
            )

    return token_ids
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # 保存这个分词器的配置
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        # 如果用户指定的特殊 token 不在词表中，就补进去
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.vocab.values():
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = token_bytes
    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens=None
    ):
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)

        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)
    def encode(self, text):
        return encode(
            text,
            self.vocab,
            self.merges,
            self.special_tokens
        )
    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    def decode(self, token_ids):
        return decode(token_ids, self.vocab)
# 读取刚才保存的训练结果
if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        Path(__file__).with_name("bpe_vocab.pkl"),
        Path(__file__).with_name("bpe_merges.pkl"),
        special_tokens=["<|endoftext|>"]
    )

    text = "cat<|endoftext|>cat!"

    token_ids = tokenizer.encode(text)
    restored = tokenizer.decode(token_ids)

    print("原始文本：", text)
    print("编码结果：", token_ids)
    print("解码结果：", restored)
    print("是否还原成功：", restored == text)
    texts = ["cat!", "cat!"]

    print("逐个接收 ID：")
    for token_id in tokenizer.encode_iterable(texts):
        print(token_id)