"""the dataset module
"""
import os
import torch

from torch.utils.data import Dataset
from PIL import Image
from .utils import make_dataset
import torchvision.transforms as transforms

DEBUG = os.environ.get("DEBUG", 0)

class ImagesDataset(Dataset):
    """ImagesDataset for pivot tuning.
    """
    def __init__(self,
                 source_root,
                 latent_root,
                 source_transform=None):
        self.source_paths = sorted(make_dataset(source_root), \
                            key = lambda x: int(os.path.basename(x[1]).split('.')[0]))
        self.latent_root = latent_root
        self.source_transform = source_transform

    def __len__(self):
        if DEBUG:
            return 100
        return min(len(self.source_paths), 200)
        # return len(self.source_paths)-100

    def __getitem__(self, index):
        _, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        return from_im, [x[0] for x in latent]

class ImagesDataset_LM(Dataset):
    """ImagesDataset for pivot tuning.
    """
    def __init__(self,
                 source_root,
                 source_transform=None):
        self.source_paths = sorted(make_dataset(source_root), \
                            key = lambda x: int(os.path.basename(x[1]).split('.')[0]))
        self.source_transform = source_transform

    def __len__(self):
        if DEBUG:
            return 100
        return min(len(self.source_paths), 1800)
        # return len(self.source_paths)-100

    def __getitem__(self, index):
        _, from_path = self.source_paths[index]
        landmarks_path = from_path.replace('/smooth/', '/landmarks_line_n_30/')

        from_im = Image.open(from_path).convert('RGB')
        lm_im = Image.open(landmarks_path).convert('RGB')

        if self.source_transform:
            from_im = self.source_transform(from_im)
            lm_im = self.source_transform(lm_im)
        mouth_im = from_im.clone()
        resize_256 = transforms.Resize((256, 256))
        mouth_im[:, 274*2:494*2, 80*2:-80*2] = lm_im[:, 274*2:494*2, 80*2:-80*2]
        mouth_im = resize_256(mouth_im)
        return from_im, mouth_im


class ImagesDataset_W(Dataset):
    """ImagesDataset for pivot tuning.
    """
    def __init__(self,
                 source_root,
                 latent_root,
                 source_transform=None):
        self.source_paths = sorted(make_dataset(source_root), \
                            key = lambda x: int(os.path.basename(x[1]).split('.')[0]))
        self.latent_root = latent_root
        self.source_transform = source_transform

    def __len__(self):
        if DEBUG:
            return 100
        return min(len(self.source_paths), 1800)
        # return len(self.source_paths)-100

    def __getitem__(self, index):
        _, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        if self.source_transform:
            from_im = self.source_transform(from_im)
        latent = torch.load(os.path.join(self.latent_root, f'{index + 1}.pt'))
        return from_im, latent
