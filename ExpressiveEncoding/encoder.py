import os
import sys
where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(where_am_i, "encoder4editing"))

import torch
import argparse
from models.psp import pSp
from models.encoders.psp_encoders import Encoder4Editing

def get_e4e_model(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net = net.to(device)
    return net, opts

class Encoder4EditingWrapper:
    def __init__(
                  self,
                  checkpoint_path
                ):
        self.net, _ = get_e4e_model(checkpoint_path)

    def __call__(
                 self,
                 x,
                 is_cars = False
                ):
        with torch.no_grad():
            codes = self.net.encoder(x)
        if self.net.opts.start_from_latent_avg:
            if codes.ndim == 2:
                codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
            else:
                codes = codes + self.net.latent_avg.repeat(codes.shape[0], 1, 1)
        if codes.shape[1] == 18 and is_cars:
            codes = codes[:, :16, :]
        return codes
