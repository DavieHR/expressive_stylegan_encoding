"""get style space latent codes script.
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import tqdm
import click
import torch
import time

from tensorboardX import SummaryWriter
from ExpressiveEncoding.train import pivot_finetuning, StyleSpaceDecoder, \
                                 stylegan_path, edict, yaml, \
                                 logger

@click.command()
@click.option('--config_path')
@click.option('--save_path')
@click.option('--resume_path', default = None)
def pivot_training(
                config_path: str,
                save_path: str,
                resume_path: str
              ):

    tensorboard = os.path.join(save_path, "tensorboard", f"{time.time()}")
    snapshots = os.path.join(save_path, "snapshots")
    os.makedirs(snapshots, exist_ok = True)

    writer = SummaryWriter(tensorboard)
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))
    resolution = config.resolution if hasattr(config, "resolution") else 1024
    decoder = StyleSpaceDecoder(stylegan_path = stylegan_path, to_resolution = resolution)
    if resume_path is not None:
        decoder.load_state_dict(torch.load(resume_path), False)
    latest_path = pivot_finetuning(config.gt_path, \
                                   config.latent_path, \
                                   snapshots, \
                                   decoder, \
                                   config.pti, \
                                   writer = writer, \
                                   epochs = config.epochs, \
                                   resolution = resolution, \
                                   batchsize = config.batchsize, \
                                   lr = config.lr \
                                   )
                                  
    logger.info(f"training finished; the lastet snapshot saved in {latest_path}")

if __name__ == '__main__':
    pivot_training()
