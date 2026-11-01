"""单条提示文本的自回归生成；复用模型，无 KV cache。"""

import math

import torch


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=256, temperature=0.8, top_p=0.9, eos_id=None, seed=42):
    if not prompt_ids or any(type(token) is not int for token in prompt_ids):
        raise ValueError("prompt_ids 必须是非空整数列表")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens 必须为非负整数")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须为有限正数")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0,1] 内")
    vocab_size = model.lm_head.weight.shape[0]
    if min(prompt_ids) < 0 or max(prompt_ids) >= vocab_size:
        raise ValueError("提示 token ID 超出模型词表范围")
    if eos_id is not None and not 0 <= eos_id < vocab_size:
        raise ValueError("EOS ID 超出模型词表范围")
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new_ids = []
    reason = "max_new_tokens"
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(ids[:, -model.context_length:])[:, -1, :]
            probabilities = torch.softmax(logits / temperature, dim=-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("采样概率出现非有限数值，请检查模型或温度")
            sorted_probs, sorted_ids = torch.sort(probabilities, dim=-1, descending=True, stable=True)
            if top_p < 1:
                keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
                sorted_probs = sorted_probs.masked_fill(~keep, 0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            position = torch.multinomial(sorted_probs, 1, generator=generator)
            token = sorted_ids.gather(-1, position)
            value = token.item()
            new_ids.append(value)
            ids = torch.cat([ids, token], dim=-1)
            if eos_id is not None and value == eos_id:
                reason = "eos"
                break
    finally:
        model.train(was_training)
    return dict(token_ids=ids[0].tolist(), generated_ids=new_ids,
                generated_token_count=len(new_ids), stop_reason=reason)
