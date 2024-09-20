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
from ExpressiveEncoding.train import W_PTI_pipeline_init, StyleSpaceDecoder, \
    stylegan_path, edict, yaml, \
    logger


@click.command()
@click.option('--config_path')
@click.option('--save_path')
@click.option('--path', default=None)
def w_pivot_training(
        config_path: str,
        save_path: str,
        path: str
):
    W_PTI_pipeline_init(config_path,save_path,path)




if __name__ == '__main__':
    w_pivot_training()
