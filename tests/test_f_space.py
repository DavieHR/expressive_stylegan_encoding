import os
import sys
sys.path.insert(0, os.getcwd())
import cv2
import torch

from ExpressiveEncoding.encoder import simpleEncoder
from ExpressiveEncoding.decoder import StyleSpaceDecoder

from ExpressiveEncoding.train import stylegan_path

def test_f_space():
    
    ss_decoder = StyleSpaceDecoder(stylegan_path, to_resolution = 1024)
    encoder = simpleEncoder()
    encoder.eval()
    image = torch.randn((1,3,512,512), dtype = torch.float32)
    w_latent = torch.randn((1, 18, 512), dtype = torch.float32)
    f = encoder(image)
    ss_latent = ss_decoder.get_style_space(w_latent)
    #gen = ss_decoder(ss_latent)
    gen = ss_decoder(ss_latent, insert_feature = {"4": f})



