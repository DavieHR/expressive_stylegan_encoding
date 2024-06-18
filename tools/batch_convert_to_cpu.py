import os
import sys
sys.path.insert(0, os.getcwd())

import re
import torch
import tqdm
import click

def convert_to_cpu(_list):
    if isinstance(_list, torch.Tensor):
        return _list.detach().cpu()
    for i, x in enumerate(_list):
        if isinstance(x, list):
            _list[i] = convert_to_cpu(x)
        elif isinstance(x, torch.Tensor):
            _list[i] = x.detach().cpu()
    return _list

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
        obj = convert_to_cpu(torch.load(_file))
        to_file = os.path.join(to_path, os.path.basename(_file))
        torch.save(obj, to_file)


if __name__ == '__main__':
    convert()


