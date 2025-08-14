# CineTrans: Learning to Generate Videos with Cinematic Transitions via Masked Diffusion Models

This repository contains the official PyTorch implementation of CineTrans, a novel framework for generating videos with controllable cinematic transitions via masked diffusion models.

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](你的论文链接) [![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://uknowsth.github.io/CineTrans/)

## 🎥 Demo
https://github.com/user-attachments/assets/6f112e2f-40e3-4347-bab8-3e08bfa9366c

## 📥 Installation
1. Clone the Repository
```
git clone https://github.com/UknowSth/CineTrans.git
cd CineTrans
```
2. Set up Environment
```
conda create -n cinetrans python==3.11.9
conda activate cinetrans

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 🤗 Checkpoint  
Download the required [model weights](url) and place them in the `ckpt/` directory.
```
ckpt/
│── stable-diffusion-v1-4/
│   ├── scheduler/
│   ├── text_encoder/
│   ├── tokenizer/  
│   │── unet/
│   └── vae_temporal_decoder/
│── checkpoint.pt
│── longclip-L.pt
```
## 🖥️ Inference  
To run the inference, use the following command:
```
python pipelines/sample.py --config configs/sample.yaml
```
Using a single A100 GPU, generating a single video takes approximately 40s. You can modify the relevant configurations and prompt in `configs/sample.yaml` to adjust the generation process.

## 🖼️ Gallery  

| ![coffee_cup](https://github.com/user-attachments/assets/c89e9462-a77b-44eb-91b6-bfba4c4c1567) | ![white_flower](https://github.com/user-attachments/assets/f5dffe7a-69da-4cc9-ba53-3549f46df904) | ![snow](https://github.com/user-attachments/assets/85b4392d-f88c-496f-a08e-b9c5f6c8354c) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Shot1:[0s,4s] Shot2:[4s,8s] | Shot1:[0s,4s] Shot2:[4s,8s] | Shot1:[0s,2.75s] Shot2:[2.75s,5.5s] Shot3:[5.5s,8s] |
| ![vintage](https://github.com/user-attachments/assets/96aa859f-e8cc-4efd-802d-417cfafcf764) | ![city_night](https://github.com/user-attachments/assets/d9e3644c-1bb3-43c6-a1dd-ea4f816c04f2) | ![sea](https://github.com/user-attachments/assets/2f80ddac-d339-4e1d-83f4-962489e2a464) |
| Shot1:[0s,2.5s] Shot2:[2.5s,5s] Shot3:[5s,8s] | Shot1:[0s,2.5s] Shot2:[2.5s,5s] Shot3:[5s,8s] | Shot1:[0s,3s] Shot2:[3s,6s] Shot3:[6s,8s] |

## 📑 BiTeX  
If you find [CineTrans](https://github.com/UknowSth/CineTrans.git) useful for your research and applications, please cite using this BibTeX:

