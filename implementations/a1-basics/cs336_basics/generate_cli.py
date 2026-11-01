"""可配置生成入口；默认加载现有 5 MB 基线，生成至 256 新 token 或首次 EOS。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import torch

from cs336_basics.decoding import generate
from cs336_basics.tokenizer_demo import Tokenizer
from cs336_basics.train_cli import ARCHITECTURE, fingerprint, write_json
from cs336_basics.transformer_lm import TransformerLM


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=folder / "train_5m_checkpoint.pt")
    parser.add_argument("--model-config", type=Path, help="无配置的旧存档所需的基线模型 JSON")
    parser.add_argument("--vocab", type=Path, default=folder / "tinystories_sample_10k_vocab.pkl")
    parser.add_argument("--merges", type=Path, default=folder / "tinystories_sample_10k_merges.pkl")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.threads <= 0 or (args.device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("线程数必须为正，设备必须可用")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_config" in checkpoint:
        if args.model_config or checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("使用存档记录的基线配置，不额外覆盖结构")
        config = checkpoint["model_config"]
        config_source = "checkpoint_metadata"
    elif args.model_config:
        config = json.loads(args.model_config.read_text(encoding="utf-8"))
        config_source = "user_supplied_legacy_baseline_config"
    elif args.checkpoint.resolve() == (folder / "train_5m_checkpoint.pt").resolve():
        config = dict(vocab_size=10000, context_length=64, d_model=64,
                      num_layers=2, num_heads=4, d_ff=128, rope_theta=10000.0)
        config_source = "known_legacy_train_5m_baseline"
    else:
        raise ValueError("旧存档缺模型配置；请提供 --model-config，仅支持 pre-norm/SwiGLU/RoPE 基线")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    if set(tokenizer.vocab) != set(range(config["vocab_size"])):
        raise ValueError("分词器 ID 范围与模型词表不一致，请使用训练时的词表和合并规则")
    model = TransformerLM(**config, device=args.device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])
    iteration = checkpoint["iteration"]
    del checkpoint
    prompt_ids = tokenizer.encode(args.prompt)
    eos_id = tokenizer.encode("<|endoftext|>")[0]
    start = perf_counter()
    result = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p, eos_id, args.seed)
    elapsed = perf_counter() - start
    result.update(dict(
        prompt=args.prompt, prompt_token_count=len(prompt_ids),
        full_text=tokenizer.decode(result["token_ids"]),
        completion=tokenizer.decode(result["generated_ids"]),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_step=iteration,
        checkpoint_sha256=fingerprint(args.checkpoint),
        vocab_sha256=fingerprint(args.vocab), merges_sha256=fingerprint(args.merges),
        model_config=config, config_source=config_source, architecture=ARCHITECTURE,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
        seed=args.seed, device=args.device, torch_version=torch.__version__,
        elapsed_seconds=elapsed,
        meets_256_tokens_or_eos=(result["generated_token_count"] >= 256 or result["stop_reason"] == "eos"),
    ))
    output = args.output_dir or folder / "generation_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录必须为空，避免覆盖已有生成结果")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "generation.json", result)
    (output / "sample.txt").write_text(result["full_text"], encoding="utf-8")
    print(result["full_text"])
    print(f"\n新 token 数：{result['generated_token_count']}，停止原因：{result['stop_reason']}")
    print("结果保存于：", output)


if __name__ == "__main__":
    main()
