import os
import sys
sys.path.insert(0, os.getcwd())
import click
import re
import cv2
import tqdm
import torch
import numpy as np


from ExpressiveEncoding.encoder import Encoder4EditingWrapper
from ExpressiveEncoding.utils import to_tensor
WHERE_AM_I = os.path.dirname(os.path.realpath(__file__))
e4e_path = os.path.join(WHERE_AM_I, "..", "ExpressiveEncoding",  "third_party/models/e4e_ffhq_encode.pt")


@click.command()
@click.option("--image_path", default = None)
@click.option("--save_path", default = None)
def get_latents(
                 image_path,
                 save_path
               ):
    e4e = Encoder4EditingWrapper(e4e_path)
    file_list = [os.path.join(image_path, x) for x in sorted(os.listdir(image_path), key = lambda y: int(''.join(re.findall('[0-9]+',y))))]
    os.makedirs(save_path, exist_ok = True)

    file_list = tqdm.tqdm(file_list)

    for i, _file in enumerate(file_list):

        image = np.float32(cv2.imread(_file) / 255.0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256,256), interpolation = cv2.INTER_CUBIC)
        image_tensor = 2 * (to_tensor(image).to("cuda") - 0.5)
        with torch.no_grad():
            latent = e4e(image_tensor)

        torch.save(
                    latent.detach().cpu(),
                    os.path.join(save_path, f"{i + 1}.pt")
                  )


if __name__ == "__main__":
    get_latents()

