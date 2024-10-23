import os
import sys
sys.path.insert(0, os.getcwd())

import torch
import click

from tqdm import tqdm

@click.command()
@click.option("--from_path", default = None)
@click.option("--to_path",  default = None)
def get_f_and_s(
                from_path: str,
                to_path: str
               ):

    s_path = os.path.join(to_path, 's.pt')
    f_path = os.path.join(to_path, 'f.pt')
    os.makedirs(to_path, exist_ok = True)

    files = [os.path.join(from_path, x) for x in sorted(os.listdir(from_path), key = lambda x: int(x.split('.')[0]))]
    files = tqdm(files)

    s = []
    f = []

    for _file in files:

        info = torch.load(_file, map_location = 'cpu')

        file_name = os.path.basename(_file)

        s += [info['pivot']]
        f += [info['f']]

    torch.save(
                s,
                s_path
              )

    torch.save(
                f,
                f_path
              )


if __name__ == "__main__":
    get_f_and_s()
