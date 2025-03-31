import os
import torch
import argparse
import torchvision

from pipeline_videogen import VideoGenPipeline

# from download import find_model
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from omegaconf import OmegaConf

import io
import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from models import get_models
from utils import save_videos_grid_tav, save_video_grid
import imageio
from diffusers.models.lora import LoRALinearLayer

from longclip_model import longclip

def find_model(model_name,type_="ema"):

    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    checkpoint = checkpoint["ema"]
    return checkpoint

def main(args):
    seed = args.seed
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet = get_models(args).to(device, dtype=torch.float16)

    state_dict = find_model(args.ckpt, type_='ema')
    # state_dict = find_model(args.ckpt,type_="model")
    unet.load_state_dict(state_dict, strict=False) # 
   
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_svd_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
    # text_encoder_one must be huge; text_encoder_two must be bigG
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device) # huge
    # tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer_2")
    # text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_path, subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)  # bigG
    vitl_model, _ = longclip.load(args.long_text_encoder_path, device=device)

    # set eval mode
    unet.eval()
    vae.eval()
    text_encoder_one.eval()
    vitl_model.eval()
    vitL_encoder = vitl_model.encode_text_full

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, 
                                                       subfolder="scheduler",
                                                       beta_start=args.beta_start, 
                                                       beta_end=args.beta_end, 
                                                       beta_schedule=args.beta_schedule,
                                                       rescale_betas_zero_snr=args.rescale_betas_zero_snr)

    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                 text_encoder=text_encoder_one, 
                                 tokenizer=tokenizer_one, 
                                 scheduler=scheduler, 
                                 unet=unet,
                                 vitL_encoder=vitL_encoder,
                                 long_tokenizer=longclip.tokenize).to(device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()
    # videogen_pipeline.enable_vae_slicing()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)

    prompt = {
        'general_caption' : "The video opens with a close-up of a delicate flower in a small pot, bathed in natural light. The camera transitions to a different angle, showcasing the flower and pot from the side, highlighting the colors and the subtle shadows cast by the light. The peaceful, calm atmosphere of the scene is emphasized.",
        'shot_info' : [32, 32], # ensure a total of 64 frames
    }

    videos = videogen_pipeline(prompt, 
        video_length=args.video_length, 
        height=args.image_size[0], 
        width=args.image_size[1], 
        num_inference_steps=args.num_sampling_steps,
        guidance_scale=args.guidance_scale).video
    imageio.mimwrite(args.save_img_path + 'video.mp4', videos[0], fps=8, quality=9)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sample.yaml")
    args = parser.parse_args()

    main(OmegaConf.load(args.config))