import os
import sys

sys.path.insert(0, os.getcwd())
import torch

def merge_pose_and_expressive(
                                pose_path,
                                expressive_path
                             ):

    """
    """

    files_length = len(os.listdir(pose_path))


    for i in range(files_length):
        pose = torch.load(os.path.join(pose_path, f"{i+1}.pt")
        attr = torch.load(os.path.join(expressive_path, f"{i+1}.pt")

        



