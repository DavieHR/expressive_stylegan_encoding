"""get style space latent codes script.
"""
import os
import sys
import re
sys.path.insert(0, os.getcwd())
import tqdm
import click
import torch
import time
import torch.multiprocessing as mp
#mp.set_start_method('spawn')

from ExpressiveEncoding.train import pivot_finetuning, StyleSpaceDecoder, \
                                 stylegan_path, edict, yaml, \
                                 logger

def kernel(
           rank, 
           world_size,
           config,
           snapshots,
           decoder,
           tensorboard,
           resolution
          ):
    
    return pivot_finetuning(
                            config.gt_path, \
                            config.latent_path, \
                            snapshots, \
                            decoder, \
                            config.pti, \
                            tensorboard = tensorboard, \
                            epochs = config.epochs, \
                            resolution = resolution, \
                            batchsize = config.batchsize, \
                            lr = config.lr, \
                            rank = rank, \
                            world_size = world_size \
                           )
@click.command()
@click.option('--config_path')
@click.option('--save_path')
@click.option('--resume_path', default = None)
@click.option('--gpus', default = 1)
def pivot_training(
                config_path: str,
                save_path: str,
                resume_path: str,
                gpus: int
              ):

    assert gpus >= 1, "expected gpu device."
    tensorboard = os.path.join(save_path, "tensorboard", f"{time.time()}")
    snapshots = os.path.join(save_path, "snapshots")
    os.makedirs(snapshots, exist_ok = True)

    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))

    resolution = config.resolution if hasattr(config, "resolution") else 1024
    decoder = StyleSpaceDecoder(stylegan_path = stylegan_path, to_resolution = resolution)
    if resume_path is not None:
        if not resume_path.endswith('pt') and not resume_path.endswith('pth'):
            folder = resume_path 
            resume_path = os.path.join(folder, sorted(os.listdir(resume_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))[-1])
            print(f"latest weight path is {resume_path}")
        decoder.load_state_dict(torch.load(resume_path), False)
    
    if gpus <= 1:
        pivot_finetuning(config.gt_path, \
                         config.latent_path, \
                         snapshots, \
                         decoder, \
                         config.pti, \
                         tensorboard = tensorboard, \
                         epochs = config.epochs, \
                         resolution = resolution, \
                         batchsize = config.batchsize, \
                         lr = config.lr \
                        )
    else:
        world_size = gpus
        mp.spawn(
                 kernel,
                 args=(
                        world_size,
                        config,
                        snapshots,
                        decoder,
                        tensorboard,
                        resolution
                        #writer
                      ),
                 nprocs=world_size,
                 join=True
                )
                                  

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    pivot_training()
