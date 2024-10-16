import os
import sys
sys.path.insert(0, os.getcwd())

import click
from copy import deepcopy
from ExpressiveEncoding.train import ( \
                                       StyleSpaceDecoder, load_model, \
                                       validate_video_gen, stylegan_path \
                                     )
                                        
@click.command()
@click.option("--save_path")
@click.option("--decoder_path")
@click.option("--latents_path")
@click.option("--face_folder_path")
def invoker(
             save_path,
             decoder_path,
             latents_path,
             face_folder_path
           ):
    """
    """
    device = "cuda:0"
    G = load_model(stylegan_path,device = device).synthesis
    ss_decoder = StyleSpaceDecoder(synthesis=deepcopy(G),device=device, to_resolution = 512)

    validate_video_gen(
                        save_path,
                        decoder_path,
                        latents_path,
                        ss_decoder,
                        1000,
                        face_folder_path,
                        512,
                        False
                       )


if __name__ == "__main__":
    invoker()

