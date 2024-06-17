"""unit test for decoder module."""
import os
import sys
import torch
import pytest
sys.path.insert(0, os.getcwd())

from ExpressiveEncoding.decoder import StyleSpaceDecoder, load_model
from ExpressiveEncoding.train import stylegan_path

@pytest.mark.load
def test_load_model():
    """test load_model
    """
    _ = load_model(stylegan_path)

@pytest.mark.decoder
def test_decoder():
    """ test for StyleSpaceDecoder Class
    """
    import cv2
    ss_decoder = StyleSpaceDecoder(stylegan_path, to_resolution = 512)
    latent_codes = torch.randn((1,18,512)).to('cuda')
    ss = ss_decoder.get_style_space(latent_codes)
    out_tensor = ss_decoder(ss, noise_mode = 'const')
    image = ((out_tensor+1) * 0.5).detach().squeeze().permute((1,2,0)).cpu().numpy()
    print(image.shape)
    cv2.imwrite("ss.png", image * 255.0)

@pytest.mark.ss
def test_from_ss_decoder():
    """ test for StyleSpaceDecoder pipeline.
    """
    import imageio
    import numpy as np
    video_path = "ss_video.mp4"
    video_writer = imageio.get_writer(video_path, fps = 25)
    ss_decoder = StyleSpaceDecoder(stylegan_path)
    latent_codes = torch.load(optimized_latent_path)
    for ss in latent_codes:
        out_tensor = ss_decoder(ss, noise_mode = 'const')
        image = ((out_tensor+1) * 0.5).detach().squeeze().permute((1,2,0)).cpu().numpy()
        image = np.clip(image, 0.0, 1.0) * 255.0
        video_writer.append_data(np.uint8(image))
