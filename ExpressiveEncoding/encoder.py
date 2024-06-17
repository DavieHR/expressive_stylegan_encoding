"""The wrapper of encoder4editting.
"""
import os
import sys
import argparse
import torch
from torch import nn
where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(where_am_i, "encoder4editing"))

from models.psp import pSp
from models.encoders import psp_encoders
from models.stylegan2.model import Generator

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

def get_e4e_model(checkpoint_path, device='cuda'):
    """get e4e model function
    """
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
    """the wrapper of encoder4editing
    """
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


class Encoder4Editing_Encoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(Encoder4Editing_Encoder, self).__init__()
        self.device = 'cuda'
        self.checkpoint_path = checkpoint_path
        # Define architecture
        self.encoder = self.set_encoder()
        # Load weights if needed
        self.load_weights()
    def load_weights(self):
        if self.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.checkpoint_path))
            ckpt = torch.load(self.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.__load_latent_avg(ckpt)


    def set_encoder(self):
        encoder = psp_encoders.Encoder4Editing(50, 'ir_se').to(self.device)
        return encoder


    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):

        codes = self.encoder(x)
        # normalize with respect to the center of an average face
        if codes.ndim == 2:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        return codes

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.device)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)

