"""配平参数量，对比 SwiGLU 与无门控 SiLU 前馈网络。"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from cs336_basics import train_rope_ablation as base
from cs336_basics.linear import Linear
from cs336_basics.swiglu import silu
from cs336_basics.train_lr_sweep import batch_hash, find_baseline
from cs336_basics.train_norm_ablation import run_case


SEED = 42
SILU_FFN_SEED = 4242
SILU_D_FF = 192


class SiLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))


def non_ffn_parameter_hash(model):
    """只比较两个版本共同拥有的非前馈网络参数，包含全部 RMSNorm。"""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".ffn." in name:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_silu(model):
    before = {name: p for name, p in model.named_parameters() if ".ffn." not in name}
    before_hash = non_ffn_parameter_hash(model)
    before_count = sum(p.numel() for p in model.parameters())
    # 单独固定替代 FFN 的随机数；不影响其他模块或外部的随机数流。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SILU_FFN_SEED)
        for block in model.layers:
            reference = block.ffn.w1.weight
            old_count = sum(p.numel() for p in block.ffn.parameters())
            replacement = SiLUFFN(reference.shape[1], SILU_D_FF,
                                   device=reference.device, dtype=reference.dtype)
            if sum(p.numel() for p in replacement.parameters()) != old_count:
                raise RuntimeError("SiLU 与 SwiGLU 的单层参数量未配平")
            block.ffn = replacement
    after = {name: p for name, p in model.named_parameters() if ".ffn." not in name}
    if list(before) != list(after) or any(before[name] is not after[name] for name in before):
        raise RuntimeError("替换意外改变了前馈网络以外的参数")
    if non_ffn_parameter_hash(model) != before_hash:
        raise RuntimeError("非前馈网络参数初值改变")
    if sum(p.numel() for p in model.parameters()) != before_count:
        raise RuntimeError("模型总参数量未配平")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="两组各训练两步检查流程")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = Path(__file__).resolve().parent
    train_data = np.load(folder / "tinystories_5m_train.npy", mmap_mode="r")
    val_data = np.load(folder / "tinystories_5m_val.npy", mmap_mode="r")
    np.random.seed(base.VALIDATION_SEED)
    val_batches = [base.get_batch(val_data, 4, 64, "cpu") for _ in range(10)]
    num_steps = 2 if args.smoke else 1000
    warmup = 1 if args.smoke else 50
    np.random.seed(SEED)
    batches = [base.get_batch(train_data, 4, 64, "cpu") for _ in range(num_steps)]
    data_hash = batch_hash(batches)
    torch.manual_seed(SEED)
    model = base.TransformerLM(**base.MODEL_CONFIG)
    initial_hash = base.parameter_hash(model)
    common_hash = non_ffn_parameter_hash(model)
    initial_loss = base.evaluate_loss(model, val_batches)
    parameter_count = sum(p.numel() for p in model.parameters())
    ffn_count = sum(p.numel() for p in model.layers[0].ffn.parameters())
    with torch.no_grad():
        before = model(val_batches[0][0])
        make_silu(model)
        alternative_initial_hash = base.parameter_hash(model)
        after = model(val_batches[0][0])
        max_initial_logit_difference = (after - before).abs().max().item()
        if not torch.isfinite(after).all() or torch.allclose(before, after):
            raise RuntimeError("替代网络输出非有限，或未观察到输出变化")
    del model, before, after

    source = baseline = None
    if not args.smoke:
        source, baseline = find_baseline(folder, len(train_data), len(val_data),
                                         initial_hash, data_hash, initial_loss)
        if not Path(baseline["checkpoint"]).is_file():
            raise RuntimeError("基线存档不存在")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{timestamp}_seed{SEED}" + ("_smoke" if args.smoke else "")
    output_dir = folder / "silu_ablation_runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = dict(
        seed=SEED, silu_ffn_seed=SILU_FFN_SEED, validation_seed=base.VALIDATION_SEED,
        base_model_config=base.MODEL_CONFIG, swiglu_d_ff=128, silu_d_ff=SILU_D_FF,
        parameter_count=parameter_count, ffn_parameters_per_layer=ffn_count,
        batch_size=4, num_steps=num_steps, total_training_tokens=num_steps * 4 * 64,
        warmup_steps=warmup, max_lr=0.003, min_lr=0.0003, weight_decay=0.01,
        betas=[0.9, 0.999], eps=1e-8, gradient_clip_norm=1.0, validation_batches=10,
        non_ffn_initial_parameter_hash=common_hash,
        base_initial_parameter_hash=initial_hash, silu_initial_parameter_hash=alternative_initial_hash,
        planned_training_batches_hash=data_hash,
        initial_output_max_absolute_difference=max_initial_logit_difference,
        ffn_initialization="Custom Linear truncated normal, std=sqrt(2/(fan_in+fan_out)); weights differ in shape and cannot be paired elementwise.",
        train_dataset_tokens=len(train_data), validation_dataset_tokens=len(val_data),
        smoke=args.smoke, torch_threads=4, torch_version=torch.__version__,
        checkpoint_loading="Instantiate base TransformerLM; make_silu(model) before loading SiLU checkpoint.",
    )
    results = []
    if args.smoke:
        record = run_case(True, 0.003, batches, val_batches, output_dir,
                           initial_hash, common_hash, data_hash, warmup,
                           case_name="swiglu", shared_hash_fn=non_ffn_parameter_hash)
        record["ffn_variant"] = "swiglu"
        results.append(record)
    else:
        curve = output_dir / "swiglu_loss.csv"
        shutil.copyfile(source.parent / "with_rope_loss.csv", curve)
        results.append(dict(
            name="swiglu", ffn_variant="swiglu", use_rmsnorm=True,
            max_lr=0.003, min_lr=0.0003, status="completed", completed_steps=1000,
            initial_val_loss=baseline["initial_val_loss"], final_val_loss=baseline["final_val_loss"],
            pre_ablation_initial_parameter_hash=initial_hash,
            shared_initial_parameter_hash=common_hash, used_training_batches_hash=data_hash,
            parameter_count=parameter_count, curve=str(curve),
            checkpoint=baseline["checkpoint"], source_summary=str(source), reused=True,
        ))
        print(f"复用 SwiGLU 基线：验证 loss={baseline['final_val_loss']:.4f}", flush=True)

    def save_summary():
        (output_dir / "silu_ablation_summary.json").write_text(
            json.dumps(dict(settings=settings, results=results), ensure_ascii=False,
                       indent=2, allow_nan=False), encoding="utf-8",
        )
    save_summary()
    record = run_case(True, 0.003, batches, val_batches, output_dir,
                       initial_hash, common_hash, data_hash, warmup,
                       model_transform=make_silu, case_name="silu",
                       shared_hash_fn=non_ffn_parameter_hash)
    record["ffn_variant"] = "silu"
    if record["actual_initial_parameter_hash"] != alternative_initial_hash:
        raise RuntimeError("实际 SiLU 前馈初值与预检查不同")
    results.append(record)
    save_summary()
    print("\n===== 参数量配平的前馈网络对比 =====", flush=True)
    for result in results:
        value = result["final_val_loss"]
        text = f"{value:.4f}" if value is not None else "未完成"
        print(f"{result['ffn_variant']}：参数量={result['parameter_count']:,}，"
              f"验证 loss={text}，状态={result['status']}", flush=True)
    print("配置、曲线及存档位置：", output_dir, flush=True)


if __name__ == "__main__":
    main()
