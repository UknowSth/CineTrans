# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

def shift_latents(latents):
    # shift latents
    latents[:,:,:-1] = latents[:,:,1:].clone()
    # add new noise to the last frame
    latents[:,:,-1] = torch.randn_like(latents[:,:,-1]) #* scheduler.init_noise_sigma
    return latents

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 mask_info,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 general_pr=False,
                 p=0,
                 return_latents=False,
                 ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            # input_prompt (`str`):
            input_prompt (`List(str)`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1, # t (81-1)/4+1 = 21
                        size[1] // self.vae_stride[1], # h
                        size[0] // self.vae_stride[2]) # w

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            # context = self.text_encoder([input_prompt], self.device) 
            context = self.text_encoder(input_prompt, self.device) # wxx 更改为多个propmt
            context_null = self.text_encoder([n_prompt]*len(input_prompt), self.device) # wxx 更改为多个propmt
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            # context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = self.text_encoder(input_prompt, torch.device('cpu')) # wxx 更改为多个propmt
            context_null = self.text_encoder([n_prompt]*len(input_prompt), torch.device('cpu')) # wxx 更改为多个propmt
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            ############## Construct Mask ##############
            p1, p2 = self.patch_size[1], self.patch_size[2]
            tt, h, w = latents[0].shape[-3], latents[0].shape[-2] // p1, latents[0].shape[-1] // p2 # wxx
            text_mask, oo = len(context), self.model.text_len
            print(f'tt={tt},h={h},w={w},text_mask={text_mask},oo={oo}')

            frame_tokens = h * w
            # mask_info = [10,11]
            all_seq_length = h * w * tt

            # general_pr = False # general prompt 是否启用
            
            if not general_pr:
                if sum(mask_info) != tt or len(mask_info) != text_mask:
                    raise ValueError("mask_info error!")
            else:
                if sum(mask_info) != tt or len(mask_info) != text_mask - 1:
                    raise ValueError("mask_info error!")

            # p =   # 设置随机为True的概率，比如10%
            premask = (torch.rand(all_seq_length, all_seq_length, device=noise[0].device) < p)
            premask = premask.to(torch.bool)

            # premask = torch.zeros(all_seq_length, all_seq_length, dtype=torch.bool).to(x.device)
            premask_text = torch.zeros(all_seq_length, text_mask*oo, dtype=torch.bool).to(noise[0].device)
            # video-video
            temp_index = 0
            # premask[temp_index: , temp_index: ] = True
            for o in mask_info:
                premask[temp_index: temp_index + o * frame_tokens, temp_index: temp_index + o * frame_tokens] = True
                temp_index += o * frame_tokens
            # video-text
            temp_index = 0
            for ooo,ppp in enumerate(mask_info):
                premask_text[temp_index: temp_index + ppp * frame_tokens, ooo * oo: (1+ooo) * oo] = True
                temp_index += ppp * frame_tokens
            if general_pr: 
                premask_text[:, (-1) * oo:] = True
           
            ###########################################
            arg_c = {'context': [context], 'seq_len': seq_len, 'premask': premask, 'premask_text': premask_text} 
            arg_null = {'context': [context_null], 'seq_len': seq_len, 'premask': premask, 'premask_text': premask_text}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)


                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        if return_latents:
            del noise
        else:
            del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # print(f'videos.shape:{videos[0].shape}')

        if self.rank == 0:
            if return_latents:
                return torch.stack(latents), videos[0]
            else:
                return videos[0]
        else:
            return None

        # return videos[0] if self.rank == 0 else None


    def slowfast_generate(self,
                 input_prompt, # 一个列表 其中包括shot_prompt和general_prompt
                 mask_info, # 当前这个片段的mask信息 会用于训练 要求返回以这个mask_info构建的mask
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 # general_pr=True, # slowfast模式下 为了激活lora的作用 general_prompt起到trigger word的作用
                 p=0,
                 # return_latents=True, # 为了能够加噪 要返回latents
                 ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            # input_prompt (`str`):
            input_prompt (`List(str)`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1, # t (81-1)/4+1 = 21
                        size[1] // self.vae_stride[1], # h
                        size[0] // self.vae_stride[2]) # w

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            # context = self.text_encoder([input_prompt], self.device) 
            context = self.text_encoder(input_prompt, self.device) # wxx 更改为多个propmt
            context_null = self.text_encoder([n_prompt]*len(input_prompt), self.device) # wxx 更改为多个propmt
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            # context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = self.text_encoder(input_prompt, torch.device('cpu')) # wxx 更改为多个propmt
            context_null = self.text_encoder([n_prompt]*len(input_prompt), torch.device('cpu')) # wxx 更改为多个propmt
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            ############## Construct Mask ##############
            p1, p2 = self.patch_size[1], self.patch_size[2]
            tt, h, w = latents[0].shape[-3], latents[0].shape[-2] // p1, latents[0].shape[-1] // p2 # wxx
            text_mask, oo = len(context), self.model.text_len
            print(f'tt={tt},h={h},w={w},text_mask={text_mask},oo={oo}')

            frame_tokens = h * w
            # mask_info = [10,11]
            all_seq_length = h * w * tt
           
            if sum(mask_info) != tt or len(mask_info) != text_mask - 1:
                raise ValueError("mask_info error!")

            # p =   # 设置随机为True的概率，比如10%
            premask = (torch.rand(all_seq_length, all_seq_length, device=noise[0].device) < p)
            premask = premask.to(torch.bool)

            # premask = torch.zeros(all_seq_length, all_seq_length, dtype=torch.bool).to(x.device)
            premask_text = torch.zeros(all_seq_length, text_mask*oo, dtype=torch.bool).to(noise[0].device)
            # video-video
            temp_index = 0
            # premask[temp_index: , temp_index: ] = True
            for o in mask_info:
                premask[temp_index: temp_index + o * frame_tokens, temp_index: temp_index + o * frame_tokens] = True
                temp_index += o * frame_tokens
            # video-text
            temp_index = 0
            for ooo,ppp in enumerate(mask_info):
                premask_text[temp_index: temp_index + ppp * frame_tokens, ooo * oo: (1+ooo) * oo] = True
                temp_index += ppp * frame_tokens
            # if general_pr: 
            premask_text[:, (-1) * oo:] = True
           
            ###########################################
            arg_c = {'context': [context], 'seq_len': seq_len, 'premask': premask, 'premask_text': premask_text} 
            arg_null = {'context': [context_null], 'seq_len': seq_len, 'premask': premask, 'premask_text': premask_text}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)


                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0: 
                videos = self.vae.decode(x0)

        # if return_latents:
        del noise
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        if self.rank == 0:
            return torch.stack(latents), videos[0], premask
        else:
            return None
    
    def fifo_onestep(self,
                    input_prompt, 
                    latents, #
                    timestep,
                    sample_scheduler,
                    size=(1280, 720),
                    frame_num=81, # 这里的frame_num并不是最终的帧数
                    shift=5.0,
                    sample_solver='unipc',
                    guide_scale=5.0,
                    n_prompt="",
                    seed=-1,
                    offload_model=True):
        
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1, # t (81-1)/4+1 = 21
                        size[1] // self.vae_stride[1], # h
                        size[0] // self.vae_stride[2]) # w

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            # context = self.text_encoder([input_prompt], self.device) 
            context = self.text_encoder(input_prompt, self.device) # wxx 更改为多个propmt
            context_null = self.text_encoder([n_prompt]*len(input_prompt), self.device) # wxx 更改为多个propmt
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            # context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = self.text_encoder(input_prompt, torch.device('cpu')) 
            context_null = self.text_encoder([n_prompt]*len(input_prompt), torch.device('cpu')) 
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            timestep = torch.tensor(timestep) # [x,x] 但是这里并没有多个batch
            self.model.to(self.device)
            noise_pred_cond = self.model.fifo_forward(
                latents, timestep, [context], seq_len)#[0]
            noise_pred_uncond = self.model.fifo_forward(
                latents, timestep, [context_null], seq_len)#[0]
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)  # B, C, F ,H ,W 

            output_latents = []
            print(f'noise_pred.shape:{noise_pred.shape}')
            for i in range(noise_pred.shape[2]):
                # if i == 0 or i == 1:
                #     print(f'noise_pred[:,:,[i]]:{noise_pred[:,:,[i]].shape}')
                #     print(f'timestep:{timestep.shape}')
                #     print(f'latents[:,:,[i]]:{latents[:,:,[i]].shape}')
                temp_x0 = sample_scheduler.step(
                    noise_pred[:,:,[i]], # .unsqueeze(0),
                    timestep[i],
                    latents[:,:,[i]],
                    return_dict=False,
                    generator=seed_g)[0]
                output_latents.append(temp_x0)
                sample_scheduler._step_index = None # 防止bug

            first_latent = output_latents[0]
            output_latents = torch.cat(output_latents, dim=2)
            print(f'output_latents.shape:{output_latents.shape}')
            print(f'first_latent.shape:{first_latent.shape}')
            

        #     if offload_model:
        #         self.model.cpu()
        #         torch.cuda.empty_cache()
        # if offload_model:
        #     gc.collect()
        #     torch.cuda.synchronize()
        # if dist.is_initialized():
        #     dist.barrier()
        return output_latents, first_latent

    def fifo_generate(self,
                    args,
                    input_prompt, 
                    latents, #
                    sample_scheduler,
                    size=(1280, 720),
                    frame_num=81, # 这里的frame_num并不是最终的帧数
                    shift=5.0,
                    sample_solver='unipc',
                    guide_scale=5.0,
                    n_prompt="",
                    seed=-1,
                    offload_model=True
                ):

        F = frame_num
        target_shape = (self.vae.model.z_dim, 
                        (F - 1) // self.vae_stride[0] + 1, # t (81-1)/4+1 = 21
                        size[1] // self.vae_stride[1], # h
                        size[0] // self.vae_stride[2]) # w

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            # context = self.text_encoder([input_prompt], self.device) 
            context = self.text_encoder(input_prompt, self.device) # wxx 更改为多个propmt
            context_null = self.text_encoder([n_prompt]*len(input_prompt), self.device) # wxx 更改为多个propmt
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            # context = self.text_encoder([input_prompt], torch.device('cpu'))
            context = self.text_encoder(input_prompt, torch.device('cpu')) 
            context_null = self.text_encoder([n_prompt]*len(input_prompt), torch.device('cpu')) 
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        @contextmanager
        def noop_no_sync():
            yield
        
        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            timesteps = sample_scheduler.timesteps
            timesteps = torch.flip(timesteps, [0]) # timestep 反序 噪声等级由小到大排序

            if args.lookahead_denoising:
                timesteps = torch.cat([torch.full((args.latent_length//2,), timesteps[0]).to(timesteps.device), timesteps])

            num_iterations = args.new_latent_length + args.queue_length - args.latent_length  #if args.version == "65x512x512" else args.new_latent_length + args.queue_length

            print(f'num_iter:{num_iterations}')

            fifo_first_latents = []
        
            for i in tqdm(range(num_iterations)):
                for rank_ in reversed(range(2 * args.num_partitions if args.lookahead_denoising else args.num_partitions)): 
                    if args.lookahead_denoising:
                        start_idx = (rank_ // 2) * args.latent_length + (rank_ % 2) * (args.latent_length // 2)
                    else:
                        start_idx = rank_ * args.latent_length
                    midpoint_idx = start_idx + args.latent_length // 2 + (rank_ % 2)
                    end_idx = start_idx + args.latent_length
                    timestep = timesteps[start_idx :end_idx]
                    input_latents = latents[:,:,start_idx:end_idx].clone()
                
                    timestep = torch.tensor(timestep) # [x,x] 但是这里并没有多个batch
                    self.model.to(self.device)
                    noise_pred_cond = self.model.fifo_forward(
                        input_latents, timestep, [context], seq_len)#[0]
                    noise_pred_uncond = self.model.fifo_forward(
                        input_latents, timestep, [context_null], seq_len)#[0]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)  # B, C, F ,H ,W 

                    output_latents = []
                    for j in range(noise_pred.shape[2]):
                        # if i == 0 or i == 1:
                        #     print(f'noise_pred[:,:,[i]]:{noise_pred[:,:,[i]].shape}')
                        #     print(f'timestep:{timestep.shape}')
                        #     print(f'latents[:,:,[i]]:{latents[:,:,[i]].shape}')
                        temp_x0 = sample_scheduler.step(
                            noise_pred[:,:,[j]], # .unsqueeze(0),
                            timestep[j],
                            input_latents[:,:,[j]],
                            return_dict=False,
                            generator=seed_g)[0]
                        output_latents.append(temp_x0)
                        sample_scheduler._step_index = None # 防止bug

                    first_latent = output_latents[0]
                    output_latents = torch.cat(output_latents, dim=2)

                # sample_scheduler._step_index = None # 避免bug
                if args.lookahead_denoising:
                    latents[:,:,midpoint_idx:end_idx] = output_latents[:,:,-(end_idx-midpoint_idx):]
                else:
                    latents[:,:,start_idx:end_idx] = output_latents
                del output_latents
            
                latents = shift_latents(latents)
                fifo_first_latents.append(first_latent)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

        return fifo_first_latents
