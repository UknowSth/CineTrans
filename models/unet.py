# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.checkpoint
from einops import repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
# from rotary_embedding_torch import RotaryEmbedding


try:
    from unet_blocks import (UNetMidBlock3DCrossAttn, 
                             get_down_block, get_up_block, 
                             CrossAttnDownBlock3D, 
                             DownBlock3D, 
                             CrossAttnUpBlock3D, 
                             UpBlock3D)
    from resnet import InflatedConv3d
    from rotary_embedding_torch_mx import RotaryEmbedding
except:
    from .unet_blocks import (UNetMidBlock3DCrossAttn, 
                             get_down_block, get_up_block, 
                             CrossAttnDownBlock3D, 
                             DownBlock3D, 
                             CrossAttnUpBlock3D, 
                             UpBlock3D)
    from .resnet import InflatedConv3d
    from .rotary_embedding_torch_mx import RotaryEmbedding

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet3DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


class UNet3DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D", 
            "CrossAttnUpBlock3D", 
            "CrossAttnUpBlock3D", 
            "CrossAttnUpBlock3D"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False, # xl false
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        encoder_hid_dim: Optional[int] = None, # xl null
        encoder_hid_dim_type: Optional[str] = None, # xl null
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None, # xl null
        dual_cross_attention: bool = False, # false
        use_linear_projection: bool = False, # xl true
        class_embed_type: Optional[str] = None, # null
        addition_embed_type: Optional[str] = None, # xl text_time
        addition_time_embed_dim: Optional[int] = None, # xl 256
        num_class_embeds: Optional[int] = None, # xl null
        upcast_attention: bool = False, # xl false
        resnet_time_scale_shift: str = "default", # xl default
        resnet_skip_time_act: bool = False, # xl false
        resnet_out_scale_factor: int = 1.0, # xl 1.0
        time_embedding_type: str = "positional", # xl positional
        time_embedding_dim: Optional[int] = None, # xl null
        time_embedding_act_fn: Optional[str] = None, # xl null
        timestep_post_act: Optional[str] = None, # xl null
        time_cond_proj_dim: Optional[int] = None, # null
        conv_in_kernel: int = 3, # xl 3
        conv_out_kernel: int = 3, # xl 3
        projection_class_embeddings_input_dim: Optional[int] = None, # 2816
        class_embeddings_concat: bool = False, # xl false
        mid_block_only_cross_attention: Optional[bool] = None, # null
        cross_attention_norm: Optional[str] = None, # null
        addition_embed_type_num_heads=64,
        use_shot_mask=False, # shot_mask
    ):
        super().__init__()

        self.sample_size = sample_size

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding)

        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional": # we are here
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        if isinstance(use_shot_mask, bool):
            use_shot_mask = [[use_shot_mask]*4,[use_shot_mask],[use_shot_mask]*4]
        # print(f'use_shot_mask:{use_shot_mask}')
        # # 判断 shot_mask 是否是一个二维布尔列表
        # elif isinstance(use_shot_mask, list) :
        #     print("shot_mask 是一个二维布尔列表")
        # else:
        #     print("shot_mask 既不是布尔值，也不是二维布尔列表")
        self.shot_mask = use_shot_mask # 二维列表

        # self.shot_embedding = nn.Embedding(2, time_embed_dim) # 类别嵌入

        # if encoder_hid_dim_type is None and encoder_hid_dim is not None:
        #     encoder_hid_dim_type = "text_proj"
        #     self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
        #     logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        # if encoder_hid_dim is None and encoder_hid_dim_type is not None:
        #     raise ValueError(
        #         f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
        #     )

        # if encoder_hid_dim_type == "text_proj":
        #     self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        # elif encoder_hid_dim_type == "text_image_proj":
        #     # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
        #     # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
        #     # case when `addition_embed_type == "text_image_proj"` (Kadinsky 2.1)`
        #     self.encoder_hid_proj = TextImageProjection(
        #         text_embed_dim=encoder_hid_dim,
        #         image_embed_dim=cross_attention_dim,
        #         cross_attention_dim=cross_attention_dim,
        #     )
        # elif encoder_hid_dim_type == "image_proj":
        #     # Kandinsky 2.2
        #     self.encoder_hid_proj = ImageProjection(
        #         image_embed_dim=encoder_hid_dim,
        #         cross_attention_dim=cross_attention_dim,
        #     )
        # elif encoder_hid_dim_type is not None:
        #     raise ValueError(
        #         f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
        #     )
        # else:
        #     self.encoder_hid_proj = None # it is None

        # # class embedding
        # if class_embed_type is None and num_class_embeds is not None:
        #     self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        # elif class_embed_type == "timestep":
        #     self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        # elif class_embed_type == "identity":
        #     self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        # elif class_embed_type == "projection":
        #     if projection_class_embeddings_input_dim is None:
        #         raise ValueError(
        #             "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
        #         )
        #     # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
        #     # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
        #     # 2. it projects from an arbitrary input dimension.
        #     #
        #     # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
        #     # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
        #     # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
        #     self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        # elif class_embed_type == "simple_projection":
        #     if projection_class_embeddings_input_dim is None:
        #         raise ValueError(
        #             "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
        #         )
        #     self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        # else:
        #     self.class_embedding = None # it is None

        if addition_embed_type == "text_time": # we need this in our situation
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
            nn.init.zeros_(self.add_embedding.linear_1.weight.data)
            nn.init.zeros_(self.add_embedding.linear_2.weight.data)
        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        # if time_embedding_act_fn is None:
        #     self.time_embed_act = None # it is None
        # else:
        #     self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool): # only_cross_attention is False
            if mid_block_only_cross_attention is None: # mid_block_only_cross_attention is None
                mid_block_only_cross_attention = only_cross_attention # Thus, mid_block_only_cross_attention is False

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int): # num_attention_heads is None
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim


        # rotary_emb = RotaryEmbedding(dim = 32,
        #                              interpolate_factor = 2.)
        rotary_emb = RotaryEmbedding(dim = 32)
        # rotary_emb = None

        # down
        output_channel = block_out_channels[0]
        down_shot_mask = use_shot_mask[0] # 
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                rotary_emb=rotary_emb,
                use_shot_mask=down_shot_mask[i],
                # use_shot_mask=use_shot_mask,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            mid_shot_mask = use_shot_mask[1] # 
            self.mid_block = UNetMidBlock3DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                rotary_emb=rotary_emb,
                use_shot_mask=mid_shot_mask[-1],
                # use_shot_mask=use_shot_mask,
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        up_shot_mask = use_shot_mask[2] 
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                rotary_emb=rotary_emb,
                use_shot_mask=up_shot_mask[i],
                # use_shot_mask=use_shot_mask,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor, # noisy_model_input
        timestep: Union[torch.Tensor, float, int], # timesteps
        encoder_hidden_states: torch.Tensor, # tokens embedding
        shot_info: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_image_num: int = 0,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        emb = emb.unsqueeze(1) # 
        # emb : [bsz, frame, dim]

        # if self.class_embedding is not None:
        #     if class_labels is None:
        #         raise ValueError("class_labels should be provided when num_class_embeds > 0")

        #     if self.config.class_embed_type == "timestep":
        #         class_labels = self.time_proj(class_labels)

        #         # `Timesteps` does not contain any weights and will always return f32 tensors
        #         # there might be better ways to encapsulate this.
        #         class_labels = class_labels.to(dtype=sample.dtype)

        #     class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

        #     if self.config.class_embeddings_concat:
        #         emb = torch.cat([emb, class_emb], dim=-1)
        #     else:
        #         emb = emb + class_emb

        # if self.config.addition_embed_type == "text_time": # we are here
        #     # SDXL - style
        #     if "text_embeds" not in added_cond_kwargs:
        #         raise ValueError(
        #             f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
        #         )
        #     text_embeds = added_cond_kwargs.get("text_embeds") # shape [2, 1280], 1280 is the hidden size of the text encoder 2
        #     # print(text_embeds.shape)
        #     # exit()
        #     # print(added_cond_kwargs) # added_cond_kwargs has text_embeds and time_ids two keys;
        #     # please note that time_ids is not timesteps.
        #     if "time_ids" not in added_cond_kwargs:
        #         raise ValueError(
        #             f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
        #         )
        #     time_ids = added_cond_kwargs.get("time_ids")
        #     # print(time_ids.shape) # shape [2, 6], 6 means what?

        #     time_embeds = self.add_time_proj(time_ids.flatten())
        #     # print(time_embeds.shape) # shape [12, 256]
        #     if self.training:
        #         time_embeds = time_embeds.reshape((text_embeds.shape[0], text_embeds.shape[1], -1))
        #     else:
        #         time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        #     # print(time_embeds.shape) # shape [2, 1536]
        #     # print('text_embeds shape', text_embeds.shape)
        #     # print('time_embeds shape', time_embeds.shape)
        #     add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        #     # print(add_embeds.shape)
        #     # print(add_embeds.shape) # shape [2, 2816]; [2, 1+use-image-num, 2816]
        #     add_embeds = add_embeds.to(emb.dtype)
        #     aug_emb = self.add_embedding(add_embeds)
        

        # if self.training:
        #     video_frame_num = sample.shape[2] - use_image_num
        #     aug_emb_video = aug_emb[:, :1, ...]
        #     aug_emb_image = aug_emb[:, 1:, ...]
        #     aug_emb_video = repeat(aug_emb_video, 'b t d -> b (t f) d', f=video_frame_num)
        #     aug_emb = torch.concat([aug_emb_video, aug_emb_image], dim=1)
        #     emb = emb.unsqueeze(1) + aug_emb if aug_emb is not None else emb # emb is timesteps; aug_emb is the combination of text_embeds and time_embeds
        # else:
        #     emb = emb + aug_emb if aug_emb is not None else emb # emb is timesteps; aug_emb is the combination of text_embeds and time_embeds
        
        emb = emb + aug_emb if aug_emb is not None else emb # emb is timesteps; aug_emb is the combination of text_embeds and time_embeds

        # if self.time_embed_act is not None: # it's none
        #     emb = self.time_embed_act(emb)

        # if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj": # self.encoder_hid_proj and self.config.encoder_hid_dim_type are both none.
        #     encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        # elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
        #     # Kadinsky 2.1 - style
        #     if "image_embeds" not in added_cond_kwargs:
        #         raise ValueError(
        #             f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
        #         )

        #     image_embeds = added_cond_kwargs.get("image_embeds")
        #     encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        # elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
        #     # Kandinsky 2.2 - style
        #     if "image_embeds" not in added_cond_kwargs:
        #         raise ValueError(
        #             f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
        #         )
        #     image_embeds = added_cond_kwargs.get("image_embeds")
        #     encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        # print(encoder_hidden_states.shape) # shape [2, 77, 2048]
        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    use_image_num=use_image_num,
                    shot_info=shot_info,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                use_image_num=use_image_num,
                shot_info=shot_info,
            )

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    use_image_num=use_image_num,
                    shot_info=shot_info,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)
    
    def forward_with_cfg(self, 
                        x, 
                        t, 
                        encoder_hidden_states = None,
                        added_cond_kwargs = None,
                        class_labels: Optional[torch.Tensor] = None,
                        cfg_scale=7.0,
                        use_fp16=False):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t, encoder_hidden_states, class_labels, added_cond_kwargs=added_cond_kwargs).sample
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :4], model_out[:, 4:]
        # eps, rest = model_out[:, :3], model_out[:, 3:] # b c f h w
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, args, subfolder=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)


        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
               "CrossAttnDownBlock3D",
               "CrossAttnDownBlock3D",
               "CrossAttnDownBlock3D",
               "DownBlock3D"
        ]

        config["mid_block_type"] = "UNetMidBlock3DCrossAttn"

        config["up_block_types"] = [
               "UpBlock3D",
               "CrossAttnUpBlock3D",
               "CrossAttnUpBlock3D",
               "CrossAttnUpBlock3D"
        ]

        config["use_shot_mask"] = args.shot_mask # temporal attention shot mask
        # # sdxl training method
        # config["addition_embed_type"] = "text_time"
        # config["addition_embed_type_num_heads"] = 64
        # config["addition_time_embed_dim"] = 256
        # config["projection_class_embeddings_input_dim"] = 2816

        from diffusers.utils import WEIGHTS_NAME # diffusion_pytorch_model.bin
        
        model = cls.from_config(config)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")
        for k, v in model.state_dict().items():
            # print(k)
            if '_temp' in k:
                state_dict.update({k: v})
            if 'add_embedding' in k:
                state_dict.update({k: v})
            if 'attn_fcross' in k: # conpy parms of attn1 to attn_fcross
                k = k.replace('attn_fcross', 'attn1')
                state_dict.update({k: state_dict[k]})
            if 'norm_fcross' in k:
                k = k.replace('norm_fcross', 'norm1')
                state_dict.update({k: state_dict[k]})
        # model.load_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        return model
 