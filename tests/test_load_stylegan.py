import os
import sys
sys.path.insert(0, os.getcwd())

import torch

from ExpressiveEncoding.decoder import load_model
from ExpressiveEncoding.train import stylegan_path

def test_load_model():

    model = load_model(stylegan_path)
    print(model)

    torch.save(
                model.state_dict(),
                "G_styleganv2_f_1024.pth"
              )


