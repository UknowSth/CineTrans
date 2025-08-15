# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]

from .globals import global_variable
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    visual=False
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        # x = flash_attn.flash_attn_varlen_func(
        (x, softmax_lse, S_dmask) = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            return_attn_probs=True,
            )#.unflatten(0, (b, lq))
        x = x.unflatten(0, (b, lq))
        # print(f'S_dmask.shape:{S_dmask.shape}')
        # exit()

        if visual:
            global global_variable
            tsp_index = global_variable // 30 # 第几个timestep
            layer_index = global_variable % 30 # 第几个layer
            # print(f'{tsp_index}_{layer_index}')        
            # print(f'Q shape:{q.shape}, K shape:{k.shape}, S_mask:{S_dmask.shape}',end='',flush=True)
            s = S_dmask[0, :, :, :]
            avg_attention = torch.mean(s, dim=0)  # 形状为 (32768, 32768)
            avg_attention = avg_attention.to(torch.float32)
            avg_attention_np = avg_attention.cpu().numpy()
            avg_attention_np[avg_attention_np < 0] = 0
            avg_attention_np = avg_attention_np[:32760,:32760]
            block_size = 1560
            reshaped = avg_attention_np.reshape(21, block_size, 21, block_size)
            mean_matrix = reshaped.mean(axis=(1, 3))

            plt.figure(figsize=(8, 8))
            sns.heatmap(mean_matrix, cmap='viridis', cbar=True)
            plt.axis('off')
            plt.title(f"Average Attention Probs {tsp_index}_{layer_index}")
            plt.savefig(f"/mnt/petrelfs/wuxiaoxue/Wan2.1/results/attnprobs/attention_probs_{tsp_index}_{layer_index}.png", dpi=300)
            plt.close()
            del S_dmask
            del softmax_lse
            global_variable += 1
            # torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # exit()

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attn_mask=None,
    visual=False,
):
    # if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
    #     return flash_attention(
    #         q=q,
    #         k=k,
    #         v=v,
    #         q_lens=q_lens,
    #         k_lens=k_lens,
    #         dropout_p=dropout_p,
    #         softmax_scale=softmax_scale,
    #         q_scale=q_scale,
    #         causal=causal,
    #         window_size=window_size,
    #         deterministic=deterministic,
    #         dtype=dtype,
    #         version=fa_version,
    #     )
    # else:
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
        )
        
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    if visual:
        global global_variable
        tsp_index = global_variable // 30 # 第几个timestep
        layer_index = global_variable % 30 # 第几个layer
        print(f'{tsp_index}_{layer_index}')

        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        # attn_weight = torch.softmax(attn_weight, dim=-1)
        max_val = attn_weight.max(dim=-1, keepdim=True).values
        attn_weight -= max_val
        torch.exp(attn_weight, out=attn_weight)
        summed = torch.sum(attn_weight, dim=-1, keepdim=True)
        attn_weight /= summed
        print(f'attn_weight.shape:{attn_weight.shape}',flush=True)

        s = attn_weight[0, :, :, :]
        avg_attention = torch.mean(s, dim=0)  # 形状为 (32768, 32768)
        avg_attention = avg_attention.to(torch.float32)
        avg_attention_np = avg_attention.cpu().numpy()
        avg_attention_np[avg_attention_np < 0] = 0
        block_size = 1560
        reshaped = avg_attention_np.reshape(21, block_size, 21, block_size)
        mean_matrix = reshaped.mean(axis=(1, 3))

        plt.figure(figsize=(8, 8))
        sns.heatmap(mean_matrix, cmap='viridis', cbar=True)
        plt.axis('off')
        plt.title(f"Average Attention Probs {tsp_index}_{layer_index}")
        plt.savefig(f"/mnt/petrelfs/wuxiaoxue/Wan2.1/results/attnprobs/attention_probs_{tsp_index}_{layer_index}.png", dpi=300)
        plt.close()
        del attn_weight
        global_variable += 1
        torch.cuda.empty_cache()
        # exit()
    
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

    out = out.transpose(1, 2).contiguous()
    return out
