import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import click

from collections import OrderedDict

@click.command()
@click.option("--path1", default = None)
@click.option("--path2", default = None)
@click.option("--to_path", default = None)
def convert(
             path1,
             path2,
             to_path
           ):

    state_dict1 = torch.load(path1)
    state_dict2 = torch.load(path2)

    new_dict = OrderedDict()


    for k, v in state_dict1.items():
        if "affine" in k:
            state_dict2[k] = state_dict1[k]

    torch.save(
                state_dict2,
                to_path
              )

if __name__ == "__main__":
    convert()

