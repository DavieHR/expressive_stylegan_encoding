import os
import click
from .train import validate_video_gen, StyleSpaceDecoder, stylegan_path, torch

@click.command()
@click.option("--save_path", default = None)
@click.option("--latest_decoder_path", default = None)
@click.option("--stage_three_path", default = None)
@click.option("--face_folder_path", default = None)
@click.option("--attribute_path", default = None)
def invoker(
             save_path,
             latest_decoder_path,
             stage_three_path,
             face_folder_path,
             attribute_path
           ):
    """
    """
    validate_video_path = os.path.join(save_path, "validate_video.mp4")
    ss_decoder = StyleSpaceDecoder(stylegan_path, to_resolution = 512)

    validate_video_gen(
                        validate_video_path,
                        latest_decoder_path,
                        stage_three_path,
                        ss_decoder,
                        1000,
                        face_folder_path,
                        attribute_path
                      )

if __name__ == "__main__":
    invoker()
