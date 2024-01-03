import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import pytest

from ExpressiveEncoding.decoder import StyleSpaceDecoder, load_model

stylegan_path = "./codes/encoder4editing/pretrained_models/ffhq.pkl"
latent_code_path = "./codes/encoder4editing/portraits/input_face1/latents.pt"
optimized_latent_path = "codes/optimized_latent/facial_edit_black_every_id.pt"

@pytest.mark.load
def test_load_model():
    G = load_model(stylegan_path)

@pytest.mark.decoder
def test_decoder():
    import cv2
    import pdb
    ss_decoder = StyleSpaceDecoder(stylegan_path)
    latent_codes = torch.load(latent_code_path)
    ss = ss_decoder.get_style_space(latent_codes[0:1])
    out_tensor = ss_decoder(ss, noise_mode = 'const')
    image = ((out_tensor+1) * 0.5).detach().squeeze().permute((1,2,0)).cpu().numpy()
    cv2.imwrite("ss.png", image * 255.0)

@pytest.mark.ss
def test_from_ss_decoder():
    import imageio
    import numpy as np
    import pdb
    video_path = "ss_video.mp4"
    video_writer = imageio.get_writer(video_path, fps = 25)
    ss_decoder = StyleSpaceDecoder(stylegan_path)
    latent_codes = torch.load(optimized_latent_path)
    for ss in latent_codes:
        out_tensor = ss_decoder(ss, noise_mode = 'const')
        image = ((out_tensor+1) * 0.5).detach().squeeze().permute((1,2,0)).cpu().numpy()
        image = np.clip(image, 0.0, 1.0) * 255.0
        video_writer.append_data(np.uint8(image))
