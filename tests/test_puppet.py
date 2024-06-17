import os
import sys
sys.path.insert(0, os.getcwd())
import pytest
import cv2

from ExpressiveEncoding.puppet import puppet

def test_puppet():
    puppet_image_path = "./results/exp005/data/puppet/smooth/0.jpg"
    latents_folder = "./results/exp006/expressive"
    config_path = "./scripts/exp004/pti.yaml"
    save_path = "./tests/puppet"
    face_folder_path = "./results/exp004/data/smooth"

    puppet( 
           puppet_image_path,
           latents_folder,
           save_path,
           config_path,
           face_folder_path
          )











