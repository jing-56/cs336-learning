"""保持参数、数据和训练设置一致，对比 pre-norm 与 post-norm。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from cs336_basics import train_rope_ablation as base
from cs336_basics.train_lr_sweep import batch_hash, find_baseline
from cs336_basics.train_norm_ablation import run_case, shared_parameter_hash


SEED = 42


class PostNormBlock(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        # 直接沿用原模块，保留参数值、名称和注册顺序，不重新初始化。
        self.ln1 = original_block.ln1
        self.ln2 = original_block.ln2
        self.attn = original_block.attn
        self.ffn = original_block.ffn

    def forward(self, x, token_positions=None):
        # 先计算更新并加回原向量，再进行归一化。
        z = self.ln1(x + self.attn(x, token_positions=token_positions))
        output = self.ln2(z + self.ffn(z))
        return output


def make_post_norm(model):
    original_parameters = dict(model.named_parameters())
    original_hash = base.parameter_hash(model)
    for index, block in enumerate(model.layers):
        model.layers[index] = PostNormBlock(block)
    new_parameters = dict(model.named_parameters())
    if list(original_parameters) != list(new_parameters):
        raise RuntimeError("变换后参数名称或顺序改变")
    if any(original_parameters[name] is not new_parameters[name] for name in original_parameters):
        raise RuntimeError("变换意外替换了参数对象")
    if base.parameter_hash(model) != original_hash:
        raise RuntimeError("变换意外修改了参数值")
    # 模型输出层前的 ln_final 保持原位置，只改变 Block 内的顺序。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="两组各训练两步，检查流程")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = Path(__file__).resolve().parent
    train_data = np.load(folder / "tinystories_5m_train.npy", mmap_mode="r")
    val_data = np.load(folder / "tinystories_5m_val.npy", mmap_mode="r")
    np.random.seed(base.VALIDATION_SEED)
    val_batches = [base.get_batch(val_data, 4, 64, "cpu") for _ in range(10)]
    np.random.seed(SEED)
    num_steps = 2 if args.smoke else 1000
    warmup = 1 if args.smoke else 50
    batches = [base.get_batch(train_data, 4, 64, "cpu") for _ in range(num_steps)]
    data_hash = batch_hash(batches)
    torch.manual_seed(SEED)
    model = base.TransformerLM(**base.MODEL_CONFIG)
    full_hash = base.parameter_hash(model)
    shared_hash = shared_parameter_hash(model)
    initial_loss = base.evaluate_loss(model, val_batches)
    parameter_count = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        before_logits = model(val_batches[0][0])
        make_post_norm(model)
        after_logits = model(val_batches[0][0])
        max_initial_logit_difference = (after_logits - before_logits).abs().max().item()
        if not torch.isfinite(after_logits).all() or torch.allclose(before_logits, after_logits):
            raise RuntimeError("变换后输出非有限，或未观察到计算顺序变化")
    del model, before_logits, after_logits
    source = baseline = None
    if not args.smoke:
        source, baseline = find_baseline(folder, len(train_data), len(val_data),
                                         full_hash, data_hash, initial_loss)
        if not Path(baseline["checkpoint"]).is_file():
            raise RuntimeError("基线存档不存在")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{SEED}"
    if args.smoke:
        run_name += "_smoke"
    output_dir = folder / "post_norm_runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = dict(
        seed=SEED, validation_seed=base.VALIDATION_SEED, model_config=base.MODEL_CONFIG,
        batch_size=4, num_steps=num_steps, total_training_tokens=num_steps * 4 * 64,
        warmup_steps=warmup, max_lr=0.003, min_lr=0.0003,
        weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8, gradient_clip_norm=1.0,
        validation_batches=10, parameter_count=parameter_count,
        initial_parameter_hash=full_hash, shared_initial_parameter_hash=shared_hash,
        planned_training_batches_hash=data_hash,
        initial_output_max_absolute_difference=max_initial_logit_difference,
        final_norm_retained=True, smoke=args.smoke, torch_threads=4,
        torch_version=torch.__version__,
        checkpoint_loading="Instantiate TransformerLM; make_post_norm(model) before loading post-norm checkpoint.",
    )
    results = []
    if args.smoke:
        pre_result = run_case(True, 0.003, batches, val_batches, output_dir,
                              full_hash, shared_hash, data_hash, warmup, case_name="pre_norm")
        pre_result["norm_placement"] = "pre"
        results.append(pre_result)
    else:
        curve = output_dir / "pre_norm_loss.csv"
        shutil.copyfile(source.parent / "with_rope_loss.csv", curve)
        results.append(dict(
            name="pre_norm", norm_placement="pre", use_rmsnorm=True,
            max_lr=0.003, min_lr=0.0003, status="completed", completed_steps=1000,
            initial_val_loss=baseline["initial_val_loss"], final_val_loss=baseline["final_val_loss"],
            pre_ablation_initial_parameter_hash=full_hash,
            shared_initial_parameter_hash=shared_hash, used_training_batches_hash=data_hash,
            parameter_count=parameter_count, curve=str(curve),
            checkpoint=baseline["checkpoint"], source_summary=str(source), reused=True,
        ))
        print(f"复用 pre-norm 基线：验证 loss={baseline['final_val_loss']:.4f}", flush=True)

    def save_summary():
        (output_dir / "post_norm_summary.json").write_text(
            json.dumps(dict(settings=settings, results=results), ensure_ascii=False,
                       indent=2, allow_nan=False), encoding="utf-8",
        )
    save_summary()
    post_result = run_case(True, 0.003, batches, val_batches, output_dir,
                           full_hash, shared_hash, data_hash, warmup,
                           model_transform=make_post_norm, case_name="post_norm")
    post_result["norm_placement"] = "post"
    results.append(post_result)
    save_summary()
    print("\n===== 归一化位置对比 =====", flush=True)
    for result in results:
        value = result["final_val_loss"]
        text = f"{value:.4f}" if value is not None else "未完成"
        print(f"{result['norm_placement']}-norm：验证 loss={text}，状态={result['status']}", flush=True)
    print("配置、曲线及存档位置：", output_dir, flush=True)


if __name__ == "__main__":
    main()
