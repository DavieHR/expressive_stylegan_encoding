"""get style space latent codes script.
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import tqdm
import click
import torch
import time
import cv2
import re
from tensorboardX import SummaryWriter
from ExpressiveEncoding.train import StyleSpaceDecoder, \
                                 stylegan_path, edict, yaml,load_model,from_tensor,Encoder4EditingWrapper,to_tensor
import numpy as np

@click.command()
@click.option('--w_pti_ckpt_path')
@click.option('--gt_path')
@click.option('--pose_path')
@click.option('--save_path')
def get_pose_imgs(w_pti_ckpt_path,gt_path,pose_path,save_path):
    from copy import deepcopy
    G = load_model(stylegan_path).synthesis
    for p in G.parameters():
        p.requires_grad = False

    # G.load_state_dict(torch.load('/data1/chenlong/online_model_set/exp_speed_v2/exp/QjXSCiOY/pti/w_snapshots/18.pth'))
    w_decoder_path = os.path.join(w_pti_ckpt_path, sorted(os.listdir(w_pti_ckpt_path),
                                                         key=lambda x: int(''.join(re.findall('[0-9]+', x))))[-1])
    print(f"latest w_decoder weight path is {w_decoder_path}")
    G.load_state_dict(torch.load(w_decoder_path))

    ss_decoder = StyleSpaceDecoder(synthesis=deepcopy(G))
    for p in ss_decoder.parameters():
        p.requires_grad = False

    files = [os.path.join(pose_path, x) for x in os.listdir(pose_path)]
    files = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    os.makedirs(save_path, exist_ok=True)

    for i, _path in enumerate(files):
        w = torch.load(_path, map_location='cuda:0')
        style_space = ss_decoder.get_style_space(w)
        gen_tensor = ss_decoder(style_space)
        image_posed = cv2.resize(from_tensor(gen_tensor) * 0.5 + 0.5, (512, 512))[:,:,::-1]* 255.0
        gt_img = cv2.imread(f'{gt_path}/{i}.jpg')
        image_posed[:280,:,:] = gt_img[:280,:,:]
        cv2.imwrite(os.path.join(save_path, f"{i}.jpg"), image_posed)




if __name__ == '__main__':
    get_pose_imgs()
