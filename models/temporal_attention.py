import torch
from torch import nn
from typing import Optional
# from rotary_embedding_torch import RotaryEmbedding
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
import torch.nn.functional as F
from einops import rearrange, repeat
import math

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

def exists(x):
    return x is not None

class CrossAttention(nn.Module):
    r"""
    copy from diffuser 0.11.1
    A cross attention layer.
    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        use_relative_position: bool = False,
    ):
        super().__init__()
        # print('num head', heads)
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        self.dim_head = dim_head
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False # No use xformers for temporal attention
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor
    
    def reshape_for_scores(self, tensor):
        # split heads and dims
        # tensor should be [b (h w)] f (d nd)
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor
    
    def same_batch_dim_to_heads(self, tensor):
        batch_size, head_size, seq_len, dim = tensor.shape # [b (h w)] nd f d
        tensor = tensor.reshape(batch_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, use_image_num=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # [b (h w)] f (nd * d)

        # print('before reshpape query shape', query.shape)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query) # [b (h w) nd] f d
        # print('after reshape query shape', query.shape)

        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # do not use xformers for temporal attention
        # # attention, what we cannot get enough of
        # if self._use_memory_efficient_attention_xformers:
        #     hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
        #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        #     hidden_states = hidden_states.to(query.dtype)
        # else:
        #     if self._slice_size is None or query.shape[0] // self._slice_size == 1:
        #         hidden_states = self._attention(query, key, value, attention_mask)
        #     else:
        #         hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        hidden_states = self._attention(query, key, value, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        # print('query shape', query.shape)
        # print('key shape', key.shape)
        # print('value shape', value.shape)

        if attention_mask is not None:
            # print('attention_mask', attention_mask.shape)
            # print('attention_scores', attention_scores.shape)
            # exit()
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        # print(attention_probs.shape)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)
        # print(attention_probs.shape)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)
        # print(hidden_states.shape)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # print(hidden_states.shape)
        # exit()
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        # print(query.shape)
        # print(key.shape)
        # print(value.shape)
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        # print(hidden_states.shape)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # print(hidden_states.shape)
        # exit()
        return hidden_states

class TemporalAttention(CrossAttention):
    def __init__(self, 
                query_dim: int,
                cross_attention_dim: Optional[int] = None,
                heads: int = 8,
                dim_head: int = 64,
                dropout: float = 0.0,
                bias=False,
                upcast_attention: bool = False,
                upcast_softmax: bool = False,
                added_kv_proj_dim: Optional[int] = None,
                norm_num_groups: Optional[int] = None,
                rotary_emb=None,
                use_shot_mask=False,
                ):
        super().__init__(query_dim, cross_attention_dim, heads, dim_head, dropout, bias, upcast_attention, upcast_softmax, added_kv_proj_dim, norm_num_groups)
        # relative time positional embeddings
        self.time_rel_pos_bias = RelativePositionBias(heads=heads, max_distance=32) # realistically will not be able to generate that many frames of video... yet
        self.rotary_emb = rotary_emb
        self.use_shot_mask = use_shot_mask
        # self.rotary_emb = RotaryEmbedding(32)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, use_image_num=None,shot_info=None):
        time_rel_pos_bias = self.time_rel_pos_bias(video_length, device=hidden_states.device)
        # time_rel_pos_bias = None
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) # [b (h w)] f (nd * d)
        dim = query.shape[-1]
        
        if self.added_kv_proj_dim is not None:
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)
            encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

            key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
            
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # Do not use xformers for temporal attention
        # attention, what we cannot get enough of
        # if self._use_memory_efficient_attention_xformers:
        #     hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
        #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        #     hidden_states = hidden_states.to(query.dtype)
        # else:
        #     if self._slice_size is None or query.shape[0] // self._slice_size == 1:
        #         hidden_states = self._attention(query, key, value, attention_mask, time_rel_pos_bias)
        #     else:
        #         hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        if self.use_shot_mask: # 启用shot mask机制 每个shot的帧只能看见所在shot中的frame
            attention_mask = shot_info.unsqueeze(2) == shot_info.unsqueeze(1)
            attention_mask = attention_mask.int().unsqueeze(1) # (batch_size, h, f, f)
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            # attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            # print(f'attention_mask1.shape:{attention_mask.shape}')
            # print(attention_mask)


        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value, shot_info, attention_mask, time_rel_pos_bias, video_length, use_image_num)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


    def _attention(self, query, key, value, shot_info=None, attention_mask=None, time_rel_pos_bias=None, video_length=None, use_image_num=None,):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        # reshape for adding time positional bais
        query = self.scale * rearrange(query, 'b f (h d) -> b h f d', h=self.heads) # d: dim_head; n: heads
        key = rearrange(key, 'b f (h d) -> b h f d', h=self.heads) # d: dim_head; n: heads
        value = rearrange(value, 'b f (h d) -> b h f d', h=self.heads) # d: dim_head; n: heads

        # torch.baddbmm only accepte 3-D tensor
        # https://runebook.dev/zh/docs/pytorch/generated/torch.baddbmm
        # attention_scores = self.scale * torch.matmul(query, key.transpose(-1, -2))
        if exists(self.rotary_emb):
            # query, _ = self.rotary_emb.rotate_queries_or_keys(query)
            # key, _ = self.rotary_emb.rotate_queries_or_keys(key)

            # if isinstance(self.embedding_module, RotaryEmbedding):
            #     print("Using RotaryEmbedding logic")
            #     # 针对 RotaryEmbedding 的逻辑
            # elif isinstance(self.embedding_module, RotaryShotEmbedding):
            #     print("Using RotaryShotEmbedding logic")

            # ## video-image mask learning
            query[:, :, :video_length, ...], _ = self.rotary_emb.rotate_queries_or_keys(query[:, :, :video_length, ...])
            key[:, :, :video_length, ...], _ = self.rotary_emb.rotate_queries_or_keys(key[:, :, :video_length, ...])
            # # add pe for image under mask
            # query[:, :, :video_length, ...], freqs = self.rotary_emb.rotate_queries_or_keys(query[:, :, :video_length, ...])
            # key[:, :, :video_length, ...], _ = self.rotary_emb.rotate_queries_or_keys(key[:, :, :video_length, ...])
            # for i in range(use_image_num):
            #     query[:, :, video_length+i: video_length+i+1, ...] = self.apply_rotary_emb(freqs[:1, ...], query[:, :, video_length+i: video_length+i+1, ...])
            #     key[:, :, video_length+i: video_length+i+1, ...] = self.apply_rotary_emb(freqs[:1, ...], key[:, :, video_length+i: video_length+i+1, ...])

        attention_scores = torch.einsum('... h i d, ... h j d -> ... h i j', query, key)
        if shot_info is not None:
            bsz = attention_scores.shape[0] // shot_info.shape[0]

        # video-image mask learning
        time_rel_pos_bias = F.pad(time_rel_pos_bias, (0, use_image_num, 0, use_image_num), 'constant', 0).to(device=time_rel_pos_bias.device, dtype=time_rel_pos_bias.dtype)
        # print(f"time_rel_pos_bias.shape:{time_rel_pos_bias.shape}")
        # print(f"attention_scores.shape:{attention_scores.shape}")
        if time_rel_pos_bias is not None:
            attention_scores = attention_scores + time_rel_pos_bias

        # bert from huggin face
        # attention_scores = attention_scores / math.sqrt(self.dim_head)

        # # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # if attention_mask is not None:
        #     # add attention mask
        #     attention_scores = attention_scores + attention_mask

        # stable value vdm 
        attention_scores = attention_scores - attention_scores.amax(dim = -1, keepdim = True).detach()

        # # Mask out future positions (causal mask)
        # mask = torch.triu(torch.ones(16, 16), diagonal=1).to(device=attention_scores.device, dtype=attention_scores.dtype) # 
        # attention_scores.masked_fill_(mask == 1, float('-inf'))

        # # # disable the fisrt frame
        # mask = torch.zeros(16, 16).to(device=attention_scores.device, dtype=attention_scores.dtype)
        # mask[:, :1] = 1
        # mask[0, 0] = 0
        # attention_scores.masked_fill_(mask == 1, float('-inf'))

        # only enable the first frame to internact with others frames
        # mask = torch.zeros(16, 16).to(device=attention_scores.device, dtype=attention_scores.dtype)
        # mask[:1, 1:] = 1
        # attention_scores.masked_fill_(mask == 1, float('-inf'))

        # video-image mask learning
        # if self.training:
        #     attention_mask = torch.zeros(video_length + use_image_num, video_length + use_image_num).to(device=attention_scores.device, dtype=attention_scores.dtype)
        #     attention_mask.fill_diagonal_(1)
        #     attention_mask[:video_length, :video_length] = 1
        #     attention_scores.masked_fill_(attention_mask == 0, float('-inf')) # 1 keep; 0 masked

        if self.use_shot_mask and attention_mask is not None:
            # attention_mask = torch.zeros(video_length + use_image_num, video_length + use_image_num).to(device=attention_scores.device, dtype=attention_scores.dtype)
            # attention_mask.fill_diagonal_(1)
            # attention_mask[:video_length, :video_length] = 1
            attention_mask = repeat(attention_mask, "b f h w -> (b r) f h w", r=bsz)
            attention_scores.masked_fill_(attention_mask == 0, float('-inf')) # 1 keep; 0 masked
            # print(f'attention_scores.shape:{attention_scores.shape}')
            # print(attention_scores)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # a = globals.global_variable
            
        # print(attention_probs[0][0])

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # print(f'attention_probs.shape:{attention_probs.shape}')
        ####################画QK图#######################
        # global global_variable
        # my_dict = {}
        # print(f'{global_variable}/',end='',flush=True)
        # attention_probs_1 = attention_probs[:bsz] # 取第一个数据对应的attn_prob
        # attention_probs_1 = attention_probs_1.mean(dim=0) # 对第一个维度求均值
        # tsp_index = global_variable // 16 # 第几个timestep
        # layer_index = global_variable % 16 # 第几个layer
        # num_heads, num_tokens = attention_probs_1.shape[0], attention_probs_1.shape[1]
        # path = f'/mnt/petrelfs/wuxiaoxue/large-video-v2/attnention_map/train1/{tsp_index}_{layer_index}.png'
        # dict_path = f'/mnt/petrelfs/wuxiaoxue/large-video-v2/attnention_map/train1/{tsp_index}_{layer_index}.pkl'
        # # fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        # attention_probs_1 = attention_probs_1.cpu().numpy()
        # # 平均矩阵（对于所有 head 求平均）
        # mean_attention_probs = np.mean(attention_probs_1, axis=0)

        # my_dict[f'{tsp_index}_{layer_index}'] = mean_attention_probs
        # import pickle
        # with open(dict_path, 'wb') as f:
        #     pickle.dump(my_dict, f)

        # plt.figure(figsize=(32, 32))
        # sns.heatmap(mean_attention_probs, cmap='viridis', cbar=True)
        # plt.axis('off')
        # plt.title(f"Average Attention Probs {tsp_index}_{layer_index}")
        # plt.savefig(path, dpi=300)
        # plt.close()

        # # 遍历所有的 attention head 绘制热图
        # for i in range(num_heads):
        #     ax = axes[i // 3, i % 3]  # 计算当前要画的子图位置
        #     sns.heatmap(attention_probs_1[i], cmap="viridis", cbar=True, ax=ax, 
        #                 xticklabels=[0, num_tokens - 1], yticklabels=[0, num_tokens - 1])
        #     ax.set_title(f"Head {i}")
        #     ax.set_xlabel("Token Index")
        #     ax.set_ylabel("Token Index")
        #     ax.set_xticks([0, num_tokens - 1])
        #     ax.set_yticks([0, num_tokens - 1])

        # # 绘制平均值的热图
        # axes[2, 2].imshow(mean_attention_probs, cmap="viridis")
        # axes[2, 2].set_title("Average QK Matrix")
        # axes[2, 2].set_xlabel("Token Index")
        # axes[2, 2].set_ylabel("Token Index")
        # axes[2, 2].set_xticks([0, num_tokens - 1])
        # axes[2, 2].set_yticks([0, num_tokens - 1])

        # # 调整子图布局
        # plt.suptitle(f"QK Similarity Matrices for timestep {tsp_index} and layer {layer_index}", fontsize=16)
        # plt.tight_layout()
        # plt.subplots_adjust(top=0.92)
        # plt.savefig(path)
        # plt.close(fig)
        # global_variable += 1
        ####################画QK图#######################

        # compute attention output 
        hidden_states = torch.einsum('... h i j, ... h j d -> ... h i d', attention_probs, value)
        hidden_states = rearrange(hidden_states, 'b h f d -> b f (h d)')
        return hidden_states
    
        ## add by xin ##
    def apply_rotary_emb(self, freqs, t, start_index = 0, scale = 1.):
        freqs = freqs.to(t)
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * freqs.cos() * scale) + (self.rotate_half(t) * freqs.sin() * scale)
        return torch.cat((t_left, t, t_right), dim = -1)
    
    def rotate_half(self, x):
        x = rearrange(x, '... (d r) -> ... d r', r = 2)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        return rearrange(x, '... d r -> ... (d r)')
    
class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j') # num_heads, num_frames, num_frames
    
# class RelativeShotPositionBias(nn.Module):
#     def __init__(
#         self,
#         heads=8,
#         num_buckets=32,
#         max_distance=128,
#     ):
#         super().__init__()
#         self.num_buckets = num_buckets
#         self.max_distance = max_distance
#         self.relative_attention_bias = nn.Embedding(num_buckets, heads)

#     @staticmethod
#     def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
#         ret = 0
#         n = -relative_position

#         num_buckets //= 2
#         ret += (n < 0).long() * num_buckets
#         n = torch.abs(n)

#         max_exact = num_buckets // 2
#         is_small = n < max_exact

#         val_if_large = max_exact + (
#             torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
#         ).long()
#         val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

#         ret += torch.where(is_small, n, val_if_large)
#         return ret

#     def get_seq_pos(self, shot_info, device):
#         # shot_info [bsz, seq_len]
#         batch_size = shot_info.shape[0]
#         N = shot_info.shape[1]
#         diff = shot_info[:, 1:] != shot_info[:, :-1]  # 比较相邻元素，得到布尔矩阵
#         diff = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=device), diff], dim=1)  # 在第一列加一个 `True`，标记每个区间的开始
#         # print(indices.int())
#         # 获取每个区间的起始位置的索引
#         start_indices = torch.arange(N, device=device, dtype = torch.long).unsqueeze(0).expand(batch_size, -1)
#         start_indices = start_indices * diff  # 只有在变化点的位置保留索引值，其他位置为 0
#         start_indices, _ = torch.cummax(start_indices, dim=1)
#         return start_indices

#     def forward(self, shot_info, device):
#         # q_pos = torch.arange(n, dtype = torch.long, device = device)
#         # k_pos = torch.arange(n, dtype = torch.long, device = device)
#         q_pos = self.get_seq_pos(shot_info, device=device)
#         k_pos = self.get_seq_pos(shot_info, device=device)
#         rel_pos = rearrange(k_pos, 'b j -> b 1 j') - rearrange(q_pos, 'b i -> b i 1')
#         rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
#         values = self.relative_attention_bias(rp_bucket)
#         return rearrange(values, 'b i j h -> b h i j')  # bsz num_heads, num_frames, num_frames