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
from ExpressiveEncoding.train_speed_pipeline_v3 import pivot_finetuning,get_s_latents, validate_video_gen,StyleSpaceDecoder, \
                                 stylegan_path, edict, yaml, \
                                 logger

@click.command()
@click.option('--config_path')
@click.option('--save_path')
@click.option('--gt_path')
@click.option('--latent_path')
@click.option('--resume_path', default = None)
def pivot_training(
                config_path: str,
                save_path: str,
                gt_path: str,
                latent_path: str,
                resume_path: str
              ):

    tensorboard = os.path.join(save_path, "tensorboard", f"{time.time()}")
    snapshots = os.path.join(save_path, "snapshots")
    os.makedirs(snapshots, exist_ok = True)
    os.makedirs(latent_path, exist_ok = True)


    writer = SummaryWriter(tensorboard)
    with open(config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))
    resolution = config.resolution if hasattr(config, "resolution") else 512
    decoder = StyleSpaceDecoder(stylegan_path = stylegan_path, to_resolution = resolution)
    if resume_path is not None:
        decoder.load_state_dict(torch.load(resume_path), False)

    if len(os.listdir(gt_path)) == len(os.listdir(latent_path)):
        pass
    else:
        get_s_latents(gt_path, decoder, myself_e4e_path=None, s_path=latent_path)

    latest_path = pivot_finetuning(gt_path, \
                                   latent_path, \
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
    validate_video_path = os.path.join(save_path, "validate_video_pti_512.mp4")
    try:
        validate_video_gen(
                            validate_video_path,
                            latest_path,
                            latent_path,
                            decoder,
                            len(os.listdir(gt_path)),
                            gt_path
                          )
        logger.info(f"validate video located in {validate_video_path}")
    except:
        logger.info(f"validate video located in {validate_video_path}, ERROR!")

if __name__ == '__main__':
    pivot_training()
