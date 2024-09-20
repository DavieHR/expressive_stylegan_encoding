import os
import torch
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
    return images

def to_tensor(x):
    return torch.from_numpy(x).permute((2,0,1)).unsqueeze(0).to(torch.float32)

def from_tensor(x):
    return x.detach().squeeze().permute((1,2,0)).cpu().numpy()

def make_train_dirs(path: str) -> dict:

    sub_dirs = [
                 "e4e",
                 "pose",
                 "facial",
                 "pti",
                 "expressive",
                 "pose_param",
                 "w",
                 "s",
                 "cache.pt",
                 "cache",
                 "e4e_s",
                 "face_info",
                 "pti_ft_512"
               ]

    sub_dirs = [os.path.join(path, x) for x in sub_dirs]

    keys = [
            "stage_one_path",
            "stage_two_path",
            "stage_three_path",
            "stage_four_path",
            "expressive_param_path",
            "stage_two_param_path",
            "w_path",
            "s_path",
            "cache_path",
            "cache_m_path",
            "stage_one_path_s",
            "face_info_path",
            "stage_four_512_path"
           ]

    sub_dirs = dict(zip(keys, sub_dirs))

    for k, v in sub_dirs.items():
        if not v.endswith(".pt"):
            os.makedirs(v, exist_ok = True)
    return sub_dirs
