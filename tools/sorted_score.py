"""get sorted scores.
"""
import os
import sys

import json
import click
from collections import OrderedDict

def get_sorted_score_frame(
                            from_path :str,
                            json_path : str,
                            to_path: str
                          ):
    """
    """

    with open(json_path) as f:
        scores = json.load(f)

    scores = dict(sorted(scores.items(), key = lambda x: x[1]['loss']))
    print(list(scores.items())[-5:])

@click.command()
@click.option('--from_path')
@click.option('--json_path')
@click.option('--to_path')
def _invoker(from_path,
             json_path,
             to_path
             ):
    return get_sorted_score_frame(from_path, json_path, to_path)

if __name__ == '__main__':
    _invoker()
