# dessurt/run.py (Corrected Version)
import os
import json
import numpy as np
import torch
from model import *
from model.loss import *
from logger import Logger
from trainer import *
from utils.saliency_qa import InputGradModel
from utils import img_f

def main(resume, config, img_path, addToConfig=None, gpu=False, do_pad=None, scale=None, do_saliency=False, default_task_token='json>', dont_output_mask=False):
    # Load checkpoint and config (overrides any passed config)
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    config = checkpoint['config']
    model = eval(config['arch'])(config['model']).eval()
    model.load_state_dict({k[7:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('module.')})
    if gpu:
        model.cuda()

    # Load and preprocess image
    img = img_f.imread(img_path, False)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    
    # Ensure 3 channels (convert grayscale to RGB)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:  # Handle RGBA
        img = img[:, :, :3]

    # Resize if needed according to target dimensions from config
    target_size = config['model']['image_size']
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    # Calculate scaling factor and compute new size in (height, width) order
    scale_factor = min(target_size[0] / img.shape[0], target_size[1] / img.shape[1])
    new_size = (int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor))
    img = img_f.resize(img, new_size)

    # Pad the image to exactly match target_size (H, W)
    pad_h = max(target_size[0] - img.shape[0], 0)
    pad_w = max(target_size[1] - img.shape[1], 0)
    img = np.pad(
        img,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
        mode='constant'
    )

    # Convert to tensor: (1, C, H, W) and normalize
    img = img.transpose(2, 0, 1)[None, ...]
    img = 1.0 - torch.from_numpy(img.astype(np.float32)) / 128.0
    if gpu:
        img = img.cuda()

    # Run inference
    with torch.no_grad():
        answer, _ = model(img, [[default_task_token]], RUN=True)
        print('\nExtracted Invoice Data:', answer)

if __name__ == '__main__':
    pass  # Handled by the main script 
