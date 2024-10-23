"""The wrapper of encoder4editting.
"""
import os
import sys
import argparse
import torch
from math import sqrt
where_am_i = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(where_am_i, "encoder4editing"))

import numpy as np
import torch.nn as nn

from models.psp import pSp


class vgg_base_module(nn.Module):
    def __init__(
                 self,
                 in_channels,
                 out_channels,
                 norm = "BatchNorm2d",
                 repeat = 1,
                 res = False,
                 **kwargs
                ):
        super().__init__()
        ksize = 3
        padding = ksize // 2
        norm = getattr(nn, norm)
        self.module = [
                        nn.Conv2d(in_channels, out_channels, ksize, stride = 2, padding = 1),
                        norm(out_channels),
                        nn.LeakyReLU(0.2),
                      ]
        for _ in range(repeat):
            self.module += [
                            nn.Conv2d(out_channels, out_channels, ksize, stride = 1, padding = 1),
                            norm(out_channels),
                           ]

        self.act = nn.LeakyReLU(0.2)

        self.res = res
        self.res_module = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        y = self.module(x)
        if self.res:
            _,_,h,w = y.shape
            x = nn.functional.interpolate(x, (h,w), mode = "nearest")
            x= self.res_module(x)
            y = (x * (1 / sqrt(2.0)) + y)
        return self.act(y)
              


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
                  checkpoint_path,
                  device='cuda:0'
                ):
        self.net, _ = get_e4e_model(checkpoint_path,device)

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

class simpleEncoder(nn.Module):
    """
       encoder arch borrow from "BDInvert: GAN Inversion for Out-of-Range Images with Geometric Transformations"
    """

    def __init__(
                  self,
                  in_channels = 3,
                  source_size = 64,
                  base_filter_num = 128,
                  target_size = 4,
                  target_filter_num = 512,
                  **kwargs
                ):
        super().__init__()
        LOG2 = lambda x: int(np.log10(x) / np.log10(2))
        level = LOG2(source_size) - LOG2(target_size)

        self.module = [nn.Conv2d(in_channels, base_filter_num, 3, 1, 1)]
        input_channels = base_filter_num
        for i in range(1, level + 1):
            output_channels = min(base_filter_num * 2 ** i, target_filter_num)

            self.module.append(vgg_base_module(input_channels, output_channels, **kwargs))
            input_channels = output_channels

        self.module += [nn.Conv2d(output_channels, target_filter_num, 1, 1, 0)]
        self.module = nn.Sequential(*self.module)
 
        self.pooling = nn.AdaptiveMaxPool2d(source_size)
        self.source_size = source_size

    def forward(self, x):
        _,_,h,w = x.shape
        if h != self.source_size or w != self.source_size:
            x = self.pooling(x)
        return self.module(x)

class simpleEncoderV2(simpleEncoder):
    def __init__(
                  self,
                  in_channels = 3,
                  source_size = 64,
                  base_filter_num = 128,
                  target_size = 4,
                  target_filter_num = 512,
                  base_code = None
                ):
        super().__init__(in_channels, source_size, base_filter_num, target_size, target_filter_num)
        self.base_code = base_code

    def forward(self, x):
        residual = super().forward(x)
        n = x.shape[0]
        return residual + (self.base_code.to(x).unsqueeze(0).repeat([n, 1, 1, 1]))
