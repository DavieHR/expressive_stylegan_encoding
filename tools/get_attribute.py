"""get attribute param.
"""
from typing import List
import os
import sys
sys.path.insert(0, os.getcwd())
import tqdm
import click
import re

import torch

from torch.utils.tensorboard import SummaryWriter
from ExpressiveEncoding.train import  StyleSpaceDecoder, stylegan_path, \
                                      alphas
                                    
alphas_split_into_region = dict(
                                 lips = alphas[:5],    #[alpha if alpha[0] in alpha_indexes[:5] else [] for alpha in alphas],
                                 chin = alphas[5:8],   #[alpha if alpha[0] in alpha_indexes[5:8] else [] for alpha in alphas],
                                 eyes = alphas[8:13],   #[alpha if alpha[0] in alpha_indexes[8:13] else [] for alpha in alphas],
                                 eyebrow = alphas[13:16], #[alpha if alpha[0] in alpha_indexes[13:16] else [] for alpha in alphas],
                                 gaze = alphas[16:] #[alpha if alpha[0] in alpha_indexes[16:] else [] for alpha in alphas],
                               )

for k, v in alphas_split_into_region.items():
    new_v = []
    for i, j in v:
        for jj in j:
            new_v += [(i, jj)]
    alphas_split_into_region[k] = new_v

lips_length = len(alphas_split_into_region["lips"])
chin_length = len(alphas_split_into_region["chin"])
eyes_length = len(alphas_split_into_region["eyes"])
eyebrow_length = len(alphas_split_into_region["eyebrow"])
gaze_length = len(alphas_split_into_region["gaze"])

masks_name = [["lips","chin"], ["eyes","eyebrow"], ["gaze"]]
region_names = ["lips"] * lips_length + ["chin"] * chin_length + ["eyes"] * eyes_length + ["eyebrow"] * eyebrow_length + ["gaze"] * gaze_length
region_num = len(masks_name)

lips_range = lips_length
chin_range = lips_range + chin_length
eyes_range = chin_range + eyes_length
eyebrow_range = eyes_range + eyebrow_length
gaze_range = eyebrow_range + gaze_length

alphas_relative_index = dict(
                              lips = list(range(lips_range)),
                              chin = list(range(lips_range, chin_range)),
                              eyes = list(range(chin_range, eyes_range)),
                              eyebrow = list(range(eyes_range, eyebrow_range)),
                              gaze = list(range(eyebrow_range, gaze_range))
                            )
def get_alpha_tensor(
                     ss_from_pose: List[torch.tensor],
                     ss_from_facial: List[torch.tensor]
                    ) -> torch.tensor:

    alpha = torch.tensor([0] * 32).type(torch.FloatTensor)

    count = 0
    # first 5 elements.
    for k, v in alphas:
        for i in v:
            alpha[count] = ss_from_facial[k][0, i] - ss_from_pose[k][0, i]
            count += 1
                
    return alpha


@click.command()
@click.option('--pose_path')
@click.option('--pose_param_path')
@click.option('--facial_path')
@click.option('--to_path')
def invoker(
             pose_path,
             pose_param_path,
             facial_path,
             to_path
           ):
    def get_files(_path):
        files = os.listdir(_path)
        assert len(files) > 0, f"no file in {_path}"
        files = sorted(files, key = lambda x: int(''.join(re.findall('[0-9]+', x))))
        return [os.path.join(_path, x) for x in files]

    poses = get_files(pose_path)
    facials = get_files(facial_path)
    ss_decoder = StyleSpaceDecoder(stylegan_path)

    length = min(len(poses), len(facials))
    for i in tqdm.tqdm(range(length)):
        pose_latent = ss_decoder.get_style_space(torch.load(poses[i]))
        facial_latent = torch.load(facials[i])
        pose_param = torch.load(os.path.join(pose_param_path, f'{i + 1}.pt'))
        alpha = get_alpha_tensor(pose_latent, facial_latent)
        torch.save(alpha, os.path.join(to_path, f"attribute_{i + 1}.pt"))

if __name__ == '__main__':

    invoker()
