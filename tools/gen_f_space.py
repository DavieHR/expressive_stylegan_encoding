import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import re
import yaml
import click
import cv2
import tqdm

import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from functools import reduce
from easydict import EasyDict as edict

from ExpressiveEncoding.encoder import simpleEncoder
from ExpressiveEncoding.utils import to_tensor, from_tensor

@click.command()
@click.option("--from_path")
@click.option("--to_path")
@click.option("--config_path")
@click.option("--model_path")
def get_f_space(
                from_path,
                to_path,
                config_path,
                model_path
                ):

    device = "cuda:0"

    images = [os.path.join(from_path, x) for x in os.listdir(from_path)]
    os.makedirs(to_path, exist_ok = True)
    
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    net_config =config.pti.net
    encoder = simpleEncoder(
                            base_filter_num = net_config.base_filter_num, \
                            source_size = net_config.source_size, \
                            target_size= net_config.target_size, \
                            target_filter_num = net_config.target_filter_num
                           )

    encoder.load_state_dict(torch.load(model_path))

    encoder.eval()
    encoder.to(device)
    images = tqdm(images)

    for x in images:
        image = ((to_tensor(np.float32(cv2.imread(x)[..., ::-1] / 255.0)) - 0.5) * 2).to(device)
        with torch.no_grad():
            f = encoder(image)       

        torch.save(
                    f.detach().cpu(),
                    os.path.join(to_path, os.path.basename(x).replace(".jpg", ".pt"))
                  )

if __name__ == "__main__":
    get_f_space()
