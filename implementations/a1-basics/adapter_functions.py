"""已实现的测试接入函数；不包含上游测试、断言或 fixtures。"""

import torch


def run_linear(d_in, d_out, weights, in_features):
    from cs336_basics.linear import Linear
    module = Linear(in_features=d_in, out_features=d_out, device=weights.device, dtype=weights.dtype)
    module.load_state_dict({'weight': weights})
    return module(in_features)


def run_embedding(vocab_size, d_model, weights, token_ids):
    from cs336_basics.embedding import Embedding
    module = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=weights.device, dtype=weights.dtype)
    module.load_state_dict({'weight': weights})
    return module(token_ids)


def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    from cs336_basics.swiglu import SwiGLU
    module = SwiGLU(d_model=d_model, d_ff=d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    module.load_state_dict({'w1.weight': w1_weight, 'w2.weight': w2_weight, 'w3.weight': w3_weight})
    return module(in_features)


def run_scaled_dot_product_attention(Q, K, V, mask=None):
    from cs336_basics.attention import scaled_dot_product_attention
    return scaled_dot_product_attention(q=Q, k=K, v=V, mask=mask)


def run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features):
    from cs336_basics.multihead_attention import MultiHeadSelfAttention
    module = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, device=q_proj_weight.device, dtype=q_proj_weight.dtype)
    module.load_state_dict({'q_proj.weight': q_proj_weight, 'k_proj.weight': k_proj_weight, 'v_proj.weight': v_proj_weight, 'output_proj.weight': o_proj_weight})
    return module(in_features)


def run_multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features, token_positions=None):
    from cs336_basics.multihead_attention import MultiHeadSelfAttention
    from cs336_basics.rope import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=in_features.device)
    module = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, device=q_proj_weight.device, dtype=q_proj_weight.dtype, rope=rope)
    module.load_state_dict({'q_proj.weight': q_proj_weight, 'k_proj.weight': k_proj_weight, 'v_proj.weight': v_proj_weight, 'output_proj.weight': o_proj_weight})
    return module(in_features, token_positions=token_positions)


def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    from cs336_basics.rope import RotaryPositionalEmbedding
    module = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    return module(in_query_or_key, token_positions)


def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    from cs336_basics.transformer_block import TransformerBlock
    module = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=max_seq_len, theta=theta, device=in_features.device, dtype=in_features.dtype)
    module.load_state_dict(weights)
    return module(in_features)


def run_transformer_lm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, weights, in_indices):
    from cs336_basics.transformer_lm import TransformerLM
    module = TransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta, device=in_indices.device, dtype=weights['token_embeddings.weight'].dtype)
    module.load_state_dict(weights)
    return module(in_indices)


def run_rmsnorm(d_model, eps, weights, in_features):
    from cs336_basics.rmsnorm import RMSNorm
    module = RMSNorm(d_model=d_model, eps=eps, device=weights.device, dtype=weights.dtype)
    module.load_state_dict({'weight': weights})
    return module(in_features)


def run_silu(in_features):
    from cs336_basics.swiglu import silu
    return silu(in_features)


def run_get_batch(dataset, batch_size, context_length, device):
    from cs336_basics.data import get_batch
    return get_batch(dataset=dataset, batch_size=batch_size, context_length=context_length, device=device)


def run_softmax(in_features, dim):
    from cs336_basics.nn_utils import softmax
    return softmax(in_features, dim=dim)


def run_cross_entropy(inputs, targets):
    from cs336_basics.nn_utils import cross_entropy
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters, max_l2_norm):
    from cs336_basics.nn_utils import gradient_clipping
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls():
    from cs336_basics.optimizer import AdamW
    return AdamW


def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    from cs336_basics.lr_schedule import get_lr_cosine_schedule
    return get_lr_cosine_schedule(it=it, max_learning_rate=max_learning_rate, min_learning_rate=min_learning_rate, warmup_iters=warmup_iters, cosine_cycle_iters=cosine_cycle_iters)


def run_save_checkpoint(model, optimizer, iteration, out):
    from cs336_basics.checkpoint import save_checkpoint
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(src, model, optimizer):
    from cs336_basics.checkpoint import load_checkpoint
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(vocab, merges, special_tokens=None):
    from cs336_basics.tokenizer_demo import Tokenizer
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    from cs336_basics.bpe import train_bpe
    return train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
