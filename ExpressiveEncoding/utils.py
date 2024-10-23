import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from .loss.FaceParsing.model import BiSeNet

WHERE_AM_I = os.path.dirname(os.path.realpath(__file__))
pretrained_models_path = None
if os.path.exists(os.path.join(WHERE_AM_I, f"third_party/models/stylegan2_ffhq.pkl")):
    pretrained_models_path = os.path.join(WHERE_AM_I, 'third_party/models')
else:
    pretrained_models_path = '/app/pretrained_models'

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

class face_parsing:
    def __init__(
                 self, 
                 path = os.path.join(f"{pretrained_models_path}", "79999_iter.pth")
                ):

        net = BiSeNet(19) 
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)
        net.eval()
        net.to("cpu")
        self.net = net
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):

        h, w = x.shape[:2]
        x = Image.fromarray(np.uint8(x))
        image = x.resize((512, 512), Image.BILINEAR)
        img = self.to_tensor(image).unsqueeze(0).to("cpu")
        out = self.net(img)[0].detach().squeeze(0).cpu().numpy().argmax(0)
        mask = np.zeros_like(out)
        for label in list(range(1,  7)) + list(range(10, 16)):
            mask[out == label] = 1
        out = cv2.resize(np.float32(mask), (w,h))
        return np.repeat(out[..., np.newaxis], 3, axis = 2)
