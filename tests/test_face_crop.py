import os
import sys
sys.path.insert(0, os.getcwd())

from ExpressiveEncoding.FaceToolsBox.crop_image import crop


def test_crop():
    image_folder = "puppet/woman"
    out_folder = "tests/woman"
    crop(image_folder, out_folder)
