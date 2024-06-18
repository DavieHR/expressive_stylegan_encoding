import os
import sys
sys.path.insert(0, os.getcwd())
import pickle
from ExpressiveEncoding.train import alphas


if __name__ == "__main__":

    region = sys.argv[1]
    save_path = sys.argv[2]

    save_path += f"_{region}.pkl"


    if region == "chin":
        alpha_regions = alphas[5:8]
        _list = []
        for _, alpha in enumerate(alpha_regions):
            l, channels = alpha
            for c in channels:
                _list += [(l,c)]

    
    with open(save_path, "wb") as f:
        pickle.dump(_list, f, protocol = 4)



