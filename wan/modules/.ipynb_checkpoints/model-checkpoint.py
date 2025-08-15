# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from typing import Dict

from .attention import flash_attention, attention

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2

def print_cuda_memory(tag=''):
    allocated = torch.cuda.memory_allocated(device=0) / 1024 ** 2  # MB
    reserved = torch.cuda.memory_reserved() / 1024 ** 2    # MB
    print(f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, attn_mask):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = attention( # wxx
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
            attn_mask=attn_mask)
        # x = flash_attention(
        #     q=rope_apply(q, grid_sizes, freqs),
        #     k=rope_apply(k, grid_sizes, freqs),
        #     v=v,
        #     k_lens=seq_lens,
        #     window_size=self.window_size)

        # output
        x = x.flatten(2)
        # print(f'x.dtype:{x.dtype}')
        # print(f'o.dtype:{self.o.weight.dtype}')
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, attn_mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        # x = flash_attention(q, k, v, k_lens=context_lens)
        x = attention(q, k, v, k_lens=context_lens, attn_mask=attn_mask) # wxx

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        attn_mask, # wxx
        attn_mask_text, # wxx
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs, attn_mask)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, attn_mask):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, attn_mask)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e, attn_mask_text)
        return x

    def fifo_forward(
        self,
        x, # [1,T,C]
        e, # [1,T,6,C] + [1,6,dim]
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        ):
        # print(f'e.shape {e.shape}')
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2) # 每片都是[1,T,1,dim] 共6片
        assert e[0].dtype == torch.float32
        repeat_size = x.shape[1] // e[0].shape[1]
        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2).repeat_interleave(repeat_size, dim=1)) + e[0].squeeze(2).repeat_interleave(repeat_size, dim=1), seq_lens, grid_sizes,
            freqs, None)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2].squeeze(2).repeat_interleave(repeat_size, dim=1)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, None)
            y = self.ffn(self.norm2(x).float() * (1 + e[4].squeeze(2).repeat_interleave(repeat_size, dim=1)) + e[3].squeeze(2).repeat_interleave(repeat_size, dim=1))
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5].squeeze(2).repeat_interleave(repeat_size, dim=1)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x

    def fifo_forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [1, L1, C] → [1, L1, 1, C] + [1, (1), 2, C] → [1, L1, 2, C]
            [B, C] → [B, 1, C] + [1, 2, C]
        """
        assert e.dtype == torch.float32
        repeat_size = x.shape[1] // e.shape[1]
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation.unsqueeze(1) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2).repeat_interleave(repeat_size, dim=1)) + e[0].squeeze(2).repeat_interleave(repeat_size, dim=1)))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 start_masked_layer=3,
                 end_masked_layer=20,):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.start_masked_layer = start_masked_layer
        self.end_masked_layer = end_masked_layer

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x, # noise
        t,
        context, # text prompt hidden states
        seq_len,
        premask=None,
        premask_text=None,
        clip_fea=None,
        y=None,
        use_gradient_checkpointing=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[List[Tensor]]):
                List of List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # tt, h, w = x[0].shape[-3], x[0].shape[-2], x[0].shape[-1] # wxx
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        ##
        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        # wxx: 注意更改以后的Context是一个双层的列表
        # context = self.text_embedding(
        #     torch.stack([
        #         torch.cat(
        #             [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        #         for u in context
        #     ]))
        context = [self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in v
                ])
            ) for v in context] # 没法凑长度一样
        # 接下来解决每个视频的shot数不一样的问题 → 直接在构建batch的时候增加"",并且在mask中提前规划好,一定可以确保shot数都相同
        context = torch.concat([u.reshape(1, -1, self.dim) for u in context], dim=0)
        print(f"++++++++++++++ context.shape:{context.shape}")

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        #################### Mask #################

        # print(f'context.shape:{context.shape}',flush=True)
        # text_mask, oo = context.shape[0], context.shape[1]
        # # print(f'text_mask:{text_mask},oo:{oo}')
        # context = context.reshape(1, text_mask*oo, -1) # wxx 多个prompt

       
        # mask layer
        # start_masked_layer = 3
        # end_masked_layer = 20
        # print(f'start:{start_masked_layer}')
        # print(f'end:{end_masked_layer}')

        # mask
        # p1, p2 = self.patch_size[1], self.patch_size[2]
        # frame_tokens = h * w
        # mask_info = [22,11]
        # all_seq_length = h * w * tt

        # general_pr = True # general prompt 是否启用

        # if not general_pr:
        #     if sum(mask_info) != tt or len(mask_info) != text_mask:
        #         raise ValueError("mask_info error!")
        # else:
        #     if sum(mask_info) != tt or len(mask_info) != text_mask - 1:
        #         raise ValueError("mask_info error!")
                
        # p = 0  # 设置随机为True的概率，比如10%
        # premask = (torch.rand(all_seq_length, all_seq_length, device=x.device) < p)
        # premask = premask.to(torch.bool)

        # # premask = torch.zeros(all_seq_length, all_seq_length, dtype=torch.bool).to(x.device)
        # premask_text = torch.zeros(all_seq_length, text_mask*oo, dtype=torch.bool).to(x.device)
        # # video-video
        # temp_index = 0
        # # premask[temp_index: , temp_index: ] = True
        # for o in mask_info:
        #     premask[temp_index: temp_index + o * frame_tokens, temp_index: temp_index + o * frame_tokens] = True
        #     temp_index += o * frame_tokens
        # # video-text
        # temp_index = 0
        # for ooo,ppp in enumerate(mask_info):
        #     premask_text[temp_index: temp_index + ppp * frame_tokens, ooo * oo: (1+ooo) * oo] = True
        #     temp_index += ppp * frame_tokens
        # if general_pr: 
        #     premask_text[:, (-1) * oo:] = True

        # print(f'premask.shape:{premask.shape}', flush=True)
        # print(f'premask_text.shape:{premask_text.shape}', flush=True)
        # print(f'context.shape:{context.shape}', flush=True)
        
        #################### Mask #################

        # # arguments
        # kwargs = dict(
        #     e=e0, # t_mod
        #     seq_lens=seq_lens,
        #     grid_sizes=grid_sizes,
        #     freqs=self.freqs, # freqs
        #     context=context, # context
        #     context_lens=context_lens,
        #     attn_mask=None,
        #     attn_mask_text=None,)

        # for ii,block in enumerate(self.blocks):
        #     if ii >= start_masked_layer and ii <= end_masked_layer:
        #         kwargs['attn_mask'] = premask
        #         kwargs['attn_mask_text'] = premask_text
        #     else:
        #         kwargs['attn_mask'] = None
        #         kwargs['attn_mask_text'] = None
        #     print_cuda_memory(f'Before {ii} BLOCK')
        #     x = block(x, **kwargs)

        ################ Gradient CKPT ##############
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for ii,block in enumerate(self.blocks):
            if ii >= self.start_masked_layer and ii <= self.end_masked_layer:
                if self.training and use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens, premask, premask_text,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens, premask, premask_text,)
            else:
                if self.training and use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens, None, None,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens, None, None,)
            # print_cuda_memory(f'Before {ii} BLOCK')

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x).float() #[u.float() for u in x]

    def fifo_forward(
        self,
        x,
        t, # [1,T]
        context,
        seq_len
    ):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding( # [T,dim]
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unsqueeze(0) # [1,T,6,dim]
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # print(f'e0.shape:{e0.shape}')
        context_lens = None
        context = [self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in v
                ])
            ) for v in context] 
        context = torch.concat([u.reshape(1, -1, self.dim) for u in context], dim=0)
        for ii,block in enumerate(self.blocks):
            x = block.fifo_forward(x, e0, seq_lens, grid_sizes, self.freqs, context, context_lens)

        # head
        x = self.head.fifo_forward(x, e.unsqueeze(0)) ### e [1, T, dim]

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x).float() #[u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

