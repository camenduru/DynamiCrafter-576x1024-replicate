from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/DynamiCrafter-hf')
sys.path.append("/content/DynamiCrafter-hf/scripts/evaluation")

import os
import argparse
import random
import time
from omegaconf import OmegaConf
import torch
import torchvision
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download
from einops import repeat
import torchvision.transforms as transforms
from utils.utils import instantiate_from_config
import numpy as np
import cv2

from funcs import (
    batch_ddim_sampling,
    load_model_checkpoint,
    get_latent_z,
    save_videos
)

def infer(image_path, prompt, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123, model=None):
    resolution = (576, 1024)
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.uint8)
    save_fps = 8
    seed_everything(seed)
    transform = transforms.Compose([
        transforms.Resize(min(resolution)),
        transforms.CenterCrop(resolution),
        ])
    torch.cuda.empty_cache()
    print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    start = time.time()
    if steps > 60:
        steps = 60 
    batch_size=1
    channels = model.model.diffusion_model.out_channels
    frames = model.temporal_length
    h, w = resolution[0] // 8, resolution[1] // 8
    noise_shape = [batch_size, channels, frames, h, w]
    # text cond
    text_emb = model.get_learned_conditioning([prompt])
    # img cond
    img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().to(model.device)
    img_tensor = (img_tensor / 255. - 0.5) * 2
    image_tensor_resized = transform(img_tensor) #3,256,256
    videos = image_tensor_resized.unsqueeze(0) # bchw
    z = get_latent_z(model, videos.unsqueeze(2)) #bc,1,hw
    img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)
    cond_images = model.embedder(img_tensor.unsqueeze(0)) ## blc
    img_emb = model.image_proj_model(cond_images)
    imtext_cond = torch.cat([text_emb, img_emb], dim=1)
    fs = torch.tensor([fs], dtype=torch.long, device=model.device)
    cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
    ## inference
    batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)
    ## b,samples,c,t,h,w
    video_path = './output.mp4'
    save_videos(batch_samples, './', filenames=['output'], fps=save_fps)
    # model = model.cpu()
    return video_path

class Predictor(BasePredictor):
    def setup(self) -> None:
        ckpt_path='/content/DynamiCrafter-hf/checkpoints/dynamicrafter_1024_v1/model.ckpt'
        config_file='/content/DynamiCrafter-hf/configs/inference_1024_v1.0.yaml'
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint']=False   
        self.model = instantiate_from_config(model_config)
        assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
        self.model = load_model_checkpoint(self.model, ckpt_path)
        self.model.eval()
        self.model = self.model.cuda()
    def predict(
        self,
        i2v_input_image: Path = Input(description="Input image"),
        i2v_input_text: str = Input(default="man fishing in a boat at sunset"),
        i2v_seed: int = Input(description="Random Seed", default=123, ge=0, le=10000),
        i2v_eta: float = Input(description="ETA", default=1.0, ge=0.0, le=1.0),
        i2v_cfg_scale: float = Input(description="CFG Scale", default=7.5, ge=1.0, le=15.0),
        i2v_steps: int = Input(description="Sampling steps", default=50, ge=1, le=60),
        i2v_motion: int = Input(description="Motion magnitude", default=4, ge=1, le=20),
    ) -> Path:
        video_path = infer(i2v_input_image, i2v_input_text, steps=i2v_steps, cfg_scale=i2v_cfg_scale, eta=i2v_eta, fs=i2v_motion, seed=i2v_seed, model=self.model)
        return Path(video_path)