"""get alignmented face.
"""
import os
import sys
sys.path.insert(0, os.getcwd())
import tqdm
import click

from ExpressiveEncoding.train import crop

@click.command()
@click.option('--from_path')
@click.option('--to_path')
def get_alignmented_face(
                          from_path: str,
                          to_path: str
                        ):
    """get alignmented face.
    """
    assert from_path.endswith('mp4'), "expected from_path postfix is mp4"
    os.makedirs(to_path, exist_ok = True)
    crop(from_path, to_path)

if __name__ == '__main__':
    get_alignmented_face()
