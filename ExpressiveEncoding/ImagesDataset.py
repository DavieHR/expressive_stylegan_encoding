import os
import torch

from torch.utils.data import Dataset
from PIL import Image
from .utils import make_dataset

DEBUG = os.environ.get("DEBUG", 0)

class ImagesDataset(Dataset):

    def __init__(self, 
                 source_root, 
                 latent_root,
                 source_transform=None):
        self.source_paths = sorted(make_dataset(source_root), key = lambda x: int(os.path.basename(x[1]).split('.')[0]))
        self.latent_root = latent_root
        self.source_transform = source_transform

    def __len__(self):
        if DEBUG:
            return 100
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        return from_im, [x[0] for x in latent]
