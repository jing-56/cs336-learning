import pickle
from heapq import heapify, heappop, heappush
import regex
from collections import Counter
from pathlib import Path
# 作业提供的预分词规则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PROGRESS_INTERVAL = 8192


class _DensePosting:
    """A fixed bitmap for pairs present in many words."""

    __slots__ = ("bits", "size")

    def __init__(self, word_count, words):
        self.bits = bytearray((word_count + 7) // 8)
        self.size = 0
        for word_id in words:
            self.add(word_id)

    def add(self, word_id):
        offset, shift = divmod(word_id, 8)
        mask = 1 << shift
        if not self.bits[offset] & mask:
            self.bits[offset] |= mask
            self.size += 1

    def remove(self, word_id):
        offset, shift = divmod(word_id, 8)
        mask = 1 << shift
        if not self.bits[offset] & mask:
            raise KeyError(word_id)
        self.bits[offset] ^= mask
        self.size -= 1

    def __iter__(self):
        for offset, bits in enumerate(self.bits):
            while bits:
                lowest = bits & -bits
                yield 8 * offset + lowest.bit_length() - 1
                bits ^= lowest


class _PairIndex:
    """Store dense postings as bitmaps and sparse postings by word ID."""

    def __init__(self, words, progress=None):
        self.postings = {}
        self.word_count = len(words)
        # Promote where a bitmap costs at most 16 bytes per member, and
        # demote at half that density to avoid repeated conversions.
        self.dense_threshold = max(1024, (self.word_count + 127) // 128)
        for word_id, tokens in enumerate(words):
            if progress is not None and word_id % PROGRESS_INTERVAL == 0:
                progress("build index", word_id, len(words))
            for pair in zip(tokens, tokens[1:]):
                self.add(pair, word_id)

    def add(self, pair, word_id):
        posting = self.postings.get(pair)
        if posting is None:
            self.postings[pair] = word_id
        elif isinstance(posting, int):
            if posting != word_id:
                self.postings[pair] = [posting, word_id]
        elif isinstance(posting, list):
            if word_id not in posting:
                posting.append(word_id)
                if len(posting) > 4:
                    self.postings[pair] = set(posting)
        else:
            posting.add(word_id)
            if isinstance(posting, set) and len(posting) >= self.dense_threshold:
                self.postings[pair] = _DensePosting(self.word_count, posting)

    def remove(self, pair, word_id):
        posting = self.postings[pair]
        if isinstance(posting, int):
            if posting != word_id:
                raise KeyError(word_id)
            del self.postings[pair]
            return
        posting.remove(word_id)
        if isinstance(posting, _DensePosting):
            if posting.size > self.dense_threshold // 2:
                return
            posting = self.postings[pair] = set(posting)
        if len(posting) == 1:
            self.postings[pair] = next(iter(posting))
        elif isinstance(posting, set) and len(posting) <= 4:
            self.postings[pair] = list(posting)

    def take_affected_words(self, pair):
        posting = self.postings.pop(pair)
        return (posting,) if isinstance(posting, int) else posting
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


def pretokenize_file_counts(input_path, special_tokens, chunk_chars=1024 * 1024):
    """Stream one-marker corpora, preserving complete pre-token boundaries.

    The unprocessed tail can grow to the longest document. The global Counter
    still grows with the number of distinct pre-tokens; this is not constant
    total memory. Other special-token configurations retain the original path.
    """
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if len(special_tokens) != 1 or not special_tokens[0]:
        text = Path(input_path).read_text(encoding="utf-8")
        return pretokenize_counts(text, special_tokens)

    marker = special_tokens[0]
    marker_pattern = regex.compile(regex.escape(marker))
    counts = Counter()
    pending = ""
    # TextIO keeps UTF-8 characters intact and normalizes newlines just like
    # the original Path.read_text() call.
    with Path(input_path).open("r", encoding="utf-8") as stream:
        while chunk := stream.read(chunk_chars):
            pending += chunk
            boundary = 0
            for match in marker_pattern.finditer(pending):
                boundary = match.end()
            if not boundary:
                continue
            counts.update(pretokenize_counts(pending[:boundary], special_tokens))
            pending = pending[boundary:]
    counts.update(pretokenize_counts(pending, special_tokens))
    return counts


class _PairCandidate:
    """A min-heap entry ordered by descending frequency, then descending pair."""

    __slots__ = ("count", "pair")

    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


# 调用函数，取得频率表
def train_bpe(input_path, vocab_size, special_tokens):
    # 读取文本，统计预分词片段
    return train_bpe_from_counts(
        pretokenize_file_counts(input_path, special_tokens), vocab_size, special_tokens,
    )


def train_bpe_from_counts(counts, vocab_size, special_tokens, *, progress=None):
    """Build the merge index from already-counted pre-tokens."""

    # 转换为字节 token 序列
    alphabet = tuple(bytes([value]) for value in range(256))
    words = []
    word_counts = []
    for word_id, (piece, count) in enumerate(counts.items()):
        if progress is not None and word_id % PROGRESS_INTERVAL == 0:
            progress("prepare words", word_id, len(counts))
        words.append(tuple(alphabet[value] for value in piece.encode("utf-8")))
        word_counts.append(count)
    del counts

    # 初始化基础词表和特殊 token
    vocab = dict(enumerate(alphabet))

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError("目标词表太小，容不下基础和特殊 token")

    merges = []
    # 建立“token 对 → 片段编号”的目录
    pair_index = _PairIndex(words, progress)

    # 初始时，对所有片段统计一次
    pair_counts = Counter()

    for word_id, tokens in enumerate(words):
        if progress is not None and word_id % PROGRESS_INTERVAL == 0:
            progress("count initial pairs", word_id, len(words))
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += word_counts[word_id]

    candidates = [_PairCandidate(count, pair) for pair, count in pair_counts.items()]
    heapify(candidates)
    merge_total = vocab_size - len(vocab)
    while len(vocab) < vocab_size and pair_counts:
        if progress is not None:
            progress("merge", len(merges), merge_total)
        candidate = heappop(candidates)
        while pair_counts.get(candidate.pair) != candidate.count:
            candidate = heappop(candidates)
        best_pair = candidate.pair
        changed_pairs = set()

        left, right = best_pair
        merged_token = left + right
        vocab[len(vocab)] = merged_token
        merges.append(best_pair)

        # 查目录：只找到包含这一对的片段
        affected_ids = pair_index.take_affected_words(best_pair)
        del pair_counts[best_pair]

        for affected_count, word_id in enumerate(affected_ids, 1):
            if progress is not None and affected_count % PROGRESS_INTERVAL == 0:
                progress("merge", len(merges), merge_total)
            old_tokens = words[word_id]
            weight = word_counts[word_id]

            new_tokens = []
            i = 0
            while i < len(old_tokens):
                if i + 1 < len(old_tokens) and old_tokens[i] == left and old_tokens[i + 1] == right:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(old_tokens[i])
                    i += 1
            new_tokens = tuple(new_tokens)
            words[word_id] = new_tokens

            # Keep the index entries that still point to this word.
            old_pairs = Counter(zip(old_tokens, old_tokens[1:]))
            new_pairs = Counter(zip(new_tokens, new_tokens[1:]))
            for pair, occurrences in old_pairs.items():
                if pair == best_pair:
                    continue
                remaining = new_pairs.get(pair, 0)
                if remaining == 0:
                    pair_index.remove(pair, word_id)
                if remaining != occurrences:
                    changed_pairs.add(pair)
                    pair_counts[pair] += (remaining - occurrences) * weight
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
            for pair, occurrences in new_pairs.items():
                if pair not in old_pairs:
                    changed_pairs.add(pair)
                    pair_counts[pair] += occurrences * weight
                    pair_index.add(pair, word_id)

        # Publish each final count once per merge. Old heap entries are skipped
        # when popped, and periodic rebuilding bounds their memory overhead.
        if len(candidates) + len(changed_pairs) > 2 * len(pair_counts) + 1024:
            candidates = [_PairCandidate(count, pair) for pair, count in pair_counts.items()]
            heapify(candidates)
        else:
            for pair in changed_pairs:
                count = pair_counts.get(pair)
                if count:
                    heappush(candidates, _PairCandidate(count, pair))

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
