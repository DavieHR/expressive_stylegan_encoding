import os
import sys
sys.path.insert(0, os.getcwd())

import re
import torch
import tqdm
import click


@click.command()
@click.option('--from_path')
@click.option('--to_path')
def convert(
            from_path: str,
            to_path: str
            ):
    
    files = [os.path.join(from_path, x) for x in os.listdir(from_path)]
    os.makedirs(to_path, exist_ok = True)
    for _file in tqdm.tqdm(files):
        obj = [x.detach().cpu() for x in torch.load(_file)]
        to_file = os.path.join(to_path, os.path.basename(_file))
        torch.save(obj, to_file)


if __name__ == '__main__':
    convert()


