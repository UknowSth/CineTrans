# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from datetime import datetime
import logging
import os
import random
import sys
import io
import argparse
import math
import warnings
from omegaconf import OmegaConf

warnings.filterwarnings('ignore')

import torch, random
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, rank: int, alpha: float = None, dropout: float = 0.0, dtype = None, device = None):
        super().__init__()
        self.orig = orig_linear
        if dtype is None:
            dtype = orig_linear.weight.dtype
        if device is None:
            device = orig_linear.weight.device
        in_sz, out_sz = orig_linear.in_features, orig_linear.out_features
        self.rank = rank
        if alpha is None:
            alpha = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # LoRA weights
        self.lora_down = nn.Linear(in_sz, rank, bias=False, device=device, dtype=dtype)
        self.lora_up = nn.Linear(rank, out_sz, bias=False, device=device, dtype=dtype)
        
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.use_lora = True
    
    def enable_lora(self):
        self.use_lora = True
    
    def disable_lora(self):
        self.use_lora = False

    def forward(self, x):
        base = self.orig(x)
        # delta = self.dropout(x) @ self.lora_down @ self.lora_up
        if self.use_lora:
            h = self.dropout(x)
            h = self.lora_down(h)  
            delta = self.lora_up(h)
            return base + self.scaling * delta
        else:
            return base

EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "flf2v-14B": {
            "prompt":
                "CG动画风格，一只蓝色的小鸟从地面起飞，煽动翅膀。小鸟羽毛细腻，胸前有独特的花纹，背景是蓝天白云，阳光明媚。镜跟随小鸟向上移动，展现出小鸟飞翔的姿态和天空的广阔。近景，仰视视角。",
            "first_frame":
                "examples/flf2v_input_first_frame.png",
            "last_frame":
                "examples/flf2v_input_last_frame.png",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40 if "i2v" in args.task else 50

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        if "flf2v" in args.task:
            args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.")
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (first frame) to generate the video from.")
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (last frame) to generate the video from.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (initialize_model_parallel,
                                             init_distributed_environment)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model, is_vl="i2v" in args.task or "flf2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"{cfg.num_heads} cannot be divided evenly by {args.ulysses_size}."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed)
            if prompt_output.status == False:
                logging.info(
                    f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V( # √
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    if args.start_masked_layer or args.end_masked_layer:
        wan_t2v.model.start_masked_layer = args.start_masked_layer
        wan_t2v.model.end_masked_layer = args.end_masked_layer


    if args.lora_path: 
        def fn_recursive(name, module):
            if hasattr(module, "q"):
                # processors[f"{name}_end"] = module
                module.q = LoRALinear(module.q, rank=args.lora_rank)
                module.k = LoRALinear(module.k, rank=args.lora_rank)
                module.v = LoRALinear(module.v, rank=args.lora_rank)
                module.o = LoRALinear(module.o, rank=args.lora_rank)
            if hasattr(module, "ffn"):
                module.ffn[0]= LoRALinear(module.ffn[0], rank=args.lora_rank)
                module.ffn[2]= LoRALinear(module.ffn[2], rank=args.lora_rank)

            for sub_name, child in module.named_children():
                fn_recursive(f"{name}.{sub_name}", child)

        for name, module in wan_t2v.model.named_children():
            fn_recursive(name, module)

        if "s3://" in args.lora_path:
            with io.BytesIO(client.get(args.lora_path)) as buffer:
                checkpoint = torch.load(buffer, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(args.lora_path, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logging.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]
        else:
            logging.info('Using model ckpt!')
            checkpoint = checkpoint["model"]

        model_dict = wan_t2v.model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
                # logger.info('Successfully Load weights from {}'.format(k))
            else:
                logging.info('Ignoring: {}'.format(k))
        logging.info('Successfully Load {}% original pretrained model weights!'.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        wan_t2v.model.load_state_dict(model_dict)
        logging.info('Successfully load model at {}!'.format(args.lora_path))


    mask_info = [11, 10]
    logging.info(
        f"Generating {'image' if 't2i' in args.task else 'video'} ...")


    video = wan_t2v.generate(
        args.prompt, 
        mask_info=mask_info,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
        general_pr=args.general_prompt,
        p=args.mask_p)


    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # formatted_prompt = args.prompt.replace(" ", "_").replace("/",
            #                                                          "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"results/{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")


if __name__ == "__main__":
    # args = _parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/t2v.yaml")

  
    args_cli = parser.parse_args()
    args_cfg = OmegaConf.load(args_cli.config)
    args_cli_dict = vars(args_cli)
    args_cli_conf = OmegaConf.create(args_cli_dict)
    args = OmegaConf.merge(args_cfg, args_cli_conf)

    # args = parser.parse_args()
    # args = OmegaConf.load(args.config)
    
    _validate_args(args)
    generate(args)
