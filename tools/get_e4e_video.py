"""get e4e video.
"""
import os
import sys
sys.path.insert(0, os.getcwd())

import click
import imageio
import cv2  

@click.command()
@click.option("--from_path", default = None)
@click.option("--to_path", default = None)
def get_video(
              from_path,
              to_path
             ):
    files = [os.path.join(from_path,x) for x in sorted(os.listdir(from_path), key = lambda x: x.split('.')[0])]

    writer = imageio.get_writer(to_path, fps = 25)
    for _file in files:
        image = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2RGB)
        writer.append_data(image)


if __name__ == "__main__":
    get_video()


