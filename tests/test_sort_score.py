import os
import sys
sys.path.insert(0, os.getcwd())

import pdb
import torch
import numpy as np
import pickle

from ExpressiveEncoding.train import alphas

def sort_scores(
                scores_path: str
               ):
    """sorted scores
    """
    scores = torch.load(scores_path)
    scores = [[(l, c, y) for c, y in enumerate(x.detach().cpu().numpy().tolist()[0])] for l, x in enumerate(scores)]

    scores_new = []
    for score in scores:
        scores_new += score
    return scores_new

if __name__ == "__main__":
    scores_path = sys.argv[1]
    pkl_path = sys.argv[2]
    assert pkl_path.endswith("pkl"), "expected pkl postfix."

    scores = sort_scores(scores_path)
    sorted_scores = sorted(scores, key = lambda x: x[2], reverse = True)
    
    sorted_scores = [x for x in sorted_scores if x[2] < 1e-4 and x[2] > 0.0]
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(sorted_scores, f, protocol = 4)

    print(sorted_scores, alphas[5:8])
