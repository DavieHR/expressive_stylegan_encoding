import os
import sys
sys.path.insert(0, os.getcwd())
import pytest
from ExpressiveEncoding.encoder import Encoder4EditingWrapper

def test_encoder():
    import cv2
    import torch
    from ExpressiveEncoding.decoder import load_model
    image_path = "tests/1.png"
    stylegan_path = "./codes/encoder4editing/pretrained_models/ffhq.pkl"
    e4e_path = "./ExpressiveEncoding/encoder4editing/e4e_ffhq_encode.pt"
    e4eWrapper = Encoder4EditingWrapper(e4e_path)
    image = cv2.imread(image_path)[...,::-1] / 255.0
    image = cv2.resize(image, (256,256), interpolation = cv2.INTER_CUBIC)

    def to_tensor(x):
        return torch.from_numpy(x).permute((2,0,1)).unsqueeze(0).to(torch.float32)

    def from_tensor(x):
        return x.detach().squeeze().permute((1,2,0)).cpu().numpy()

    image_tensor = 2 * (to_tensor(image) - 0.5)
    latent_code = e4eWrapper(image_tensor.to("cuda:0"))
    G = load_model(stylegan_path).synthesis
    imgs = G(latent_code)
    imgs = from_tensor(imgs) * 0.5 + 0.5
    cv2.imwrite("output.png", imgs[...,::-1] * 255.0)
