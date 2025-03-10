import os
import json
import logging
import argparse
import torch
from model import *
from model.loss import *
from logger import Logger
from trainer import *
import math
from collections import defaultdict
import pickle
import warnings
from utils.saliency_qa import InputGradModel
from utils import img_f
from skimage import future

def main(resume, config, img_path, addToConfig=None, gpu=False, do_pad=None, scale=None, do_saliency=False, default_task_token='json>', dont_output_mask=False):
    np.random.seed(1234)
    torch.manual_seed(1234)
    no_mask_qs = ['fli:', 'fna:', 're~', 'l~', 'v~', 'mm~', 'mk>', 'natural_q~', 'json>', 'json~', 'linkdown-text~', 'read_block>']
    remove_qs = ['rm>', 'mlm>', 'mm~', 'mk>']
    
    # Load checkpoint
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    config = checkpoint['config']
    for key in config.keys():
        if 'pretrained' in key:
            config[key] = None

    config['optimizer_type'] = "none"
    config['trainer']['use_learning_schedule'] = False
    config['trainer']['swa'] = False
    config['cuda'] = gpu
    config['gpu'] = gpu

    # Load model
    state_dict = checkpoint['state_dict']
    new_state_dict = {key[7:]: value for key, value in state_dict.items() if key.startswith('module.')}
    model = eval(config['arch'])(config['model'])
    model.load_state_dict(new_state_dict)
    model.eval()
    if gpu:
        model = model.cuda()

    # Image preprocessing
    img = img_f.imread(img_path, False)
    if img is None:
        raise ValueError(f"Failed to load image at {img_path}")
    
    if img.max() <= 1:
        img *= 255

    # Handle grayscale images
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)  # Convert to 3-channel grayscale

    # Resize/pad image
    do_pad = config['model']['image_size']
    if isinstance(do_pad, int):
        do_pad = (do_pad, do_pad)
    
    if img.shape[0] != do_pad[0] or img.shape[1] != do_pad[1]:
        diff_x = do_pad[1] - img.shape[1]
        diff_y = do_pad[0] - img.shape[0]
        p_img = np.zeros((do_pad[0], do_pad[1], 3), dtype=np.uint8)  # Ensure 3 channels
        p_img[
            diff_y // 2: diff_y // 2 + img.shape[0],
            diff_x // 2: diff_x // 2 + img.shape[1]
        ] = img
        img = p_img

    # Convert to tensor
    img = img.transpose([2, 0, 1])[None, ...]  # Shape: (1, C, H, W)
    img = 1.0 - torch.from_numpy(img.astype(np.float32)) / 128.0
    if gpu:
        img = img.cuda()

    # Use default query (json>) to extract invoice data
    question = default_task_token

    # Create dummy masks (no user interaction)
    mask = torch.zeros_like(img)
    rm_mask = torch.zeros_like(img)
    in_img = torch.cat((img * (1 - rm_mask), mask.to(img.device)), dim=1)

    # Run inference
    with torch.no_grad():
        answer, _ = model(in_img, [[question]], RUN=True)
        print('\nExtracted Invoice Data:', answer)
