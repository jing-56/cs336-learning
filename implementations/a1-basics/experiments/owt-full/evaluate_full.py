"""Score contiguous validation windows with a saved CS336 baseline model."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.data import get_batch
from cs336_basics.train_cli import (
    ARCHITECTURE, TrainConfig, TransformerLM, evaluate, fingerprint, load_tokens, write_json,
)


@torch.inference_mode()
def score(model, data, start, stop, context, device):
    count = stop - start
    if count % context:
        if count >= context:
            raise ValueError("Score the final short window separately.")
        context = count
    x = torch.from_numpy(np.array(data[start:stop], dtype=np.int64)).reshape(-1, context).to(device)
    y = torch.from_numpy(np.array(data[start + 1:stop + 1], dtype=np.int64)).to(device)
    logits = model(x)
    losses = F.cross_entropy(logits.flatten(0, 1), y, reduction="none")
    return losses.sum(dtype=torch.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--report-every", type=int, default=256)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if min(args.batch_size, args.report_every) <= 0 or (args.max_targets is not None and args.max_targets <= 0):
        parser.error("Batch size, report interval and optional target limit must be positive.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Expected a baseline training checkpoint (format 1).")
    config = TrainConfig(**checkpoint["config"])
    torch.set_num_threads(config.threads)
    validation_sha = fingerprint(args.validation)
    if validation_sha != checkpoint["data_hashes"]["val"]:
        raise ValueError("Validation data does not match the checkpoint.")
    data = load_tokens(args.validation, config)
    context = config.context_length
    total = len(data) - 1
    if args.max_targets is not None:
        total = min(total, args.max_targets)
    complete_stop = total - total % context
    identity = dict(
        checkpoint_sha256=fingerprint(args.checkpoint), validation_sha256=validation_sha,
        checkpoint_step=checkpoint["iteration"], validation_tokens=len(data), planned_targets=total,
        context_length=context, batch_size=args.batch_size, device=args.device,
        torch_version=str(torch.__version__), dtype="float32",
        evaluation_policy="Nonoverlapping input windows; score targets starting at index 1 once, resetting context each window.",
    )
    previous = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else None
    if previous is not None and any(previous.get(k) != v for k, v in identity.items()):
        raise ValueError("Output belongs to a different run; use a different path.")
    if previous and previous["status"] == "completed":
        print(json.dumps(previous, indent=2), flush=True)
        return
    model = TransformerLM(**config.model_config(), device=args.device).eval()
    model.load_state_dict(checkpoint["model"])
    if previous:
        fixed = previous["fixed_batch_loss"]
        fixed_fused = previous["fixed_batch_fused_loss"]
    else:
        np.random.seed(config.val_seed)
        batches = [get_batch(data, config.eval_batch_size, context, args.device) for _ in range(config.eval_batches)]
        fixed = evaluate(model, batches)
        with torch.inference_mode():
            fixed_fused = sum(
                F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in batches
            ) / len(batches)
        del batches
        if not math.isclose(fixed, fixed_fused, rel_tol=0, abs_tol=1e-5):
            raise RuntimeError(f"Cross-entropy implementations disagree: {fixed} vs {fixed_fused}.")
    print(f"step={checkpoint['iteration']:,}, fixed loss={fixed:.8f}, eval loss on same batches={fixed_fused:.8f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cursor = previous["evaluated_targets"] if previous else 0
    nll = previous["nll_sum"] if previous else 0.0
    prior_seconds = previous["evaluation_seconds"] if previous else 0.0
    if not (0 <= cursor <= complete_stop and cursor % context == 0 and math.isfinite(nll)):
        raise ValueError("Invalid resume state.")
    start = perf_counter()
    batch_targets = args.batch_size * context
    partial_nll = torch.zeros((), dtype=torch.float64, device=args.device)
    for batch, lo in enumerate(range(cursor, total, batch_targets), start=1):
        hi = min(lo + batch_targets, complete_stop) if lo < complete_stop else total
        partial_nll += score(model, data, lo, hi, context, args.device)
        if hi == complete_stop and total > complete_stop:
            partial_nll += score(model, data, complete_stop, total, context, args.device)
            hi = total
        if batch % args.report_every == 0 or hi == total:
            nll += partial_nll.item()
            if not math.isfinite(nll):
                raise RuntimeError("Non-finite loss.")
            partial_nll.zero_()
            elapsed = prior_seconds + perf_counter() - start
            rate = hi / elapsed
            result = dict(
                identity, status="completed" if hi == total else "running", evaluated_targets=hi,
                nll_sum=nll, mean_loss=nll / hi, perplexity=math.exp(nll / hi),
                final_window_targets=total % context, fixed_batch_loss=fixed,
                fixed_batch_fused_loss=fixed_fused, fixed_batch_targets=config.eval_batch_size * config.eval_batches * context,
                evaluation_seconds=elapsed,
            )
            if args.device == "cuda":
                result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            write_json(args.output, result)
            print(f"{hi:,}/{total:,} ({hi / total:.1%}) loss={nll / hi:.6f} "
                  f"{rate:,.0f} tokens/s ETA={(total - hi) / rate / 60:.1f} min", flush=True)
        if hi == total:
            break


if __name__ == "__main__":
    main()
