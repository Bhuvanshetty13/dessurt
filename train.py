# dessurt/run.py (Revised for 2-channel conversion)
import os
import json
import numpy as np
import torch
import cv2
from model import *
from model.loss import *
from logger import Logger
from trainer import *
from utils.saliency_qa import InputGradModel

def main(resume, config, img_path, addToConfig=None, gpu=False, do_pad=None, scale=None, do_saliency=False, default_task_token='json>', dont_output_mask=False):
    # Load checkpoint and configuration (overrides any passed config)
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    config = checkpoint['config']
    model = eval(config['arch'])(config['model']).eval()
    model.load_state_dict({k[7:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('module.')})
    if gpu:
        model.cuda()

    # Load image using OpenCV
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Drop alpha channel if present
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # Get target size from config (assumed to be (height, width))
    target_size = config['model']['image_size']
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    target_height, target_width = target_size

    # Calculate scaling factor: maintain aspect ratio
    scale_factor = min(target_height / img.shape[0], target_width / img.shape[1])
    # cv2.resize expects (width, height)
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    img = cv2.resize(img, new_size)

    # Pad image to exactly match the target size (height, width)
    pad_h = max(target_height - img.shape[0], 0)
    pad_w = max(target_width - img.shape[1], 0)
    img = np.pad(
        img,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
        mode='constant'
    )

    # --- Convert to 2-channel image ---
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # shape: (H, W)
    gray = gray.astype(np.float32)
    
    # Normalize: original approach was 1.0 - gray/128.0
    channel1 = 1.0 - gray / 128.0
    # Complementary channel (inversion)
    channel2 = 1.0 - channel1  # equivalent to gray/128.0

    # Stack to create a 2-channel image: shape (2, H, W)
    img = np.stack([channel1, channel2], axis=0)
    # Add batch dimension: shape becomes (1, 2, H, W)
    img = img[None, ...]
    # Debug: print the image shape
    print("Image tensor shape after conversion:", img.shape)

    # Convert numpy array to torch tensor
    img = torch.from_numpy(img)
    if gpu:
        img = img.cuda()

    # Run inference
    with torch.no_grad():
        answer, _ = model(img, [[default_task_token]], RUN=True)
        print('\nExtracted Invoice Data:', answer)

if __name__ == '__main__':
    pass  # Handled by the main script 
