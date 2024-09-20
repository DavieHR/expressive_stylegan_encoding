"""Expressive total pipeline module.
"""
import os
from typing import Callable, Union, List
from functools import reduce
import cv2
import torch
import yaml
import imageio
import pdb
import re
import warnings
import time
import numpy as np
import torch.distributed as dist

from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from easydict import EasyDict as edict
from DeepLog import logger, Timer
from DeepLog.logger import logging
from torchvision import transforms
from PIL import Image

from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

current_dir = os.getcwd()

from .pose_edit_with_flow import PoseEdit
from .decoder import StyleSpaceDecoder, load_model
from .encoder import Encoder4EditingWrapper
from .FaceToolsBox.alignment import get_detector, infer, \
                             get_euler_angle, get_landmarks_from_mediapipe_results, need_to_warp

from .FaceToolsBox.crop_image import crop
from .ImagesDataset import ImagesDataset, ImagesDatasetV2, ImagesDatasetV3, ImagesDatasetW
from .loss import LossRegisterBase
from .loss.FaceParsing.model import BiSeNet
from .loss import LPIPS

from .utils import to_tensor, from_tensor, make_train_dirs

points = [(10, 338),(338, 297),(297, 332),
          (332, 284),(284, 251),(251, 389),
          (389, 356),(356, 454),(454, 323),
          (323, 361),(361, 288),(288, 397),
          (397, 365),(365, 379),(379, 378),
          (378, 400),(400, 377),(377, 152),
          (152, 148),(148, 176),(176, 149),
          (149, 150),(150, 136),(136, 172),
          (172, 58),(58, 132),(132, 93),
          (93, 234),(234, 127),(127, 162),
          (162, 21),(21, 54),(54, 103),
          (103, 67),(67, 109),(109, 10)]

WHERE_AM_I = os.path.dirname(os.path.realpath(__file__))
pretrained_models_path = None
if os.path.exists(os.path.join(WHERE_AM_I, f"third_party/models/stylegan2_ffhq.pkl")):
    pretrained_models_path = 'third_party/models'
else:
    pretrained_models_path = '/app/pretrained_models'

stylegan_path = os.path.join(WHERE_AM_I, f"{pretrained_models_path}/stylegan2_ffhq.pkl")
e4e_path = os.path.join(WHERE_AM_I, f"{pretrained_models_path}/e4e_ffhq_encode.pt")
DEBUG = os.environ.get("DEBUG", 0)
DEBUG = True if DEBUG in [True, 'True', 'TRUE', 'true', 1] else False
VERBOSE = os.environ.get("VERBOSE", False)
VERBOSE = True if VERBOSE in [True, 'True', 'TRUE', 'true', 1] else False

if VERBOSE or DEBUG:
    logger.setLevel(logging.DEBUG)

alpha_indexes = [
                 6, 11, 8, 14, 15, # represent Mouth
                 #5, 6, 8, # Chin/Jaw
                 8, 11, 14, # Chin/Jaw
                 9, 11, 12, 14, 17, # Eyes
                 8, 9 , 11, # Eyebrows
                 9 # Gaze
                ]

alpha_S_indexes = [
                    [113, 202, 214, 259, 378, 501],
                    [6, 41, 78, 86, 313, 361, 365],
                    [17, 387],
                    [12],
                    [45],
                    #[50, 505],[131],[390],
                    [122],[78],[43, 455],
                    [63],
                    [257],
                    [82, 414],
                    [239],
                    [28],
                    [6, 28],
                    [30],
                    [320],
                    [409]
                  ]

alphas = list(zip(alpha_indexes, alpha_S_indexes))
alphas_split_into_region = dict(
                                 lips = alphas[:5],    #[alpha if alpha[0] in alpha_indexes[:5] else [] for alpha in alphas],
                                 chin = alphas[5:8],   #[alpha if alpha[0] in alpha_indexes[5:8] else [] for alpha in alphas],
                                 eyes = alphas[8:13],   #[alpha if alpha[0] in alpha_indexes[8:13] else [] for alpha in alphas],
                                 eyebrow = alphas[13:16], #[alpha if alpha[0] in alpha_indexes[13:16] else [] for alpha in alphas],
                                 gaze = alphas[16:] #[alpha if alpha[0] in alpha_indexes[16:] else [] for alpha in alphas],
                               )

for k, v in alphas_split_into_region.items():
    new_v = []
    for i, j in v:
        for jj in j:
            new_v += [(i, jj)]
    alphas_split_into_region[k] = new_v

lips_length = len(alphas_split_into_region["lips"])
chin_length = len(alphas_split_into_region["chin"])
eyes_length = len(alphas_split_into_region["eyes"])
eyebrow_length = len(alphas_split_into_region["eyebrow"])
gaze_length = len(alphas_split_into_region["gaze"])

masks_name = [["lips","chin"], ["eyes","eyebrow"], ["gaze"]]
region_names = ["lips"] * lips_length + ["chin"] * chin_length + ["eyes"] * eyes_length + ["eyebrow"] * eyebrow_length + ["gaze"] * gaze_length
region_num = len(masks_name)
region_name = ["mouth_group", "eye_group", "gaze_group"]

lips_range = lips_length
chin_range = lips_range + chin_length
eyes_range = chin_range + eyes_length
eyebrow_range = eyes_range + eyebrow_length
gaze_range = eyebrow_range + gaze_length

alphas_relative_index = dict(
                              lips = list(range(lips_range)),
                              chin = list(range(lips_range, chin_range)),
                              eyes = list(range(chin_range, eyes_range)),
                              eyebrow = list(range(eyes_range, eyebrow_range)),
                              gaze = list(range(eyebrow_range, gaze_range))
                            )

output_copy_region="[[274,494,80,432]]"
soft_mask_region="[[340,494,130,-130],[274,340,130,-130]]"
regions = eval(soft_mask_region)
output_copy_region = eval(output_copy_region)

facial_alpha_tensor_pre = None
yaw_to_optim_pre = None
pitch_to_optim_pre = None

where_am_i = os.path.dirname(os.path.realpath(__file__))
class face_parsing:
    def __init__(self, path = os.path.join(f"{pretrained_models_path}", "79999_iter.pth")):

        net = BiSeNet(19) 
        state_dict = torch.load(path)
        net.load_state_dict(state_dict)
        net.eval()
        net.to("cuda:0")
        self.net = net
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):

        h, w = x.shape[:2]
        x = Image.fromarray(np.uint8(x))
        image = x.resize((512, 512), Image.BILINEAR)
        img = self.to_tensor(image).unsqueeze(0).to("cuda:0")
        out = self.net(img)[0].detach().squeeze(0).cpu().numpy().argmax(0)
        mask = np.zeros_like(out)
        for label in list(range(1,  7)) + list(range(10, 16)):
            mask[out == label] = 1
        out = cv2.resize(np.float32(mask), (w,h))
        return out[..., np.newaxis]

# instance face parse

#face_parse = face_parsing()

# get dlatent with masked
def get_masked_dlatent(tmp_alpha_tensor, dlatents):
    dlatents_with_masked = [dlatent.clone().repeat(32, 1) for dlatent in dlatents]
    for index, (k, i) in enumerate(alphas):
        #r_i = alpha_relative_index[index]
        r_i = index
        latents = dlatents_with_masked[k]
        latents[r_i, i] = latents[r_i, i] + tmp_alpha_tensor[r_i]
        dlatents_with_masked[k] = latents
    return dlatents_with_masked

def get_mask_by_region():
    _mask  = np.zeros((512,512,3), np.float32)
    _mask_dilate = _mask.copy()
    pad = 20
    for region in output_copy_region:
        y1,y2,x1,x2 = region
        _mask[y1:y2,x1:x2,:] = 1
        _mask_dilate[y1:y1 + pad, x1:x2, :] = 1
        _mask_dilate[y2 - pad :y2, x1:x2, :] = 1
        _mask_dilate[y1:y2, x1 : x1 + pad, :] = 1
        _mask_dilate[y1:y2, x2 - pad : x2, :] = 1

    return _mask, _mask_dilate

def get_up_bottom_mask():
    pad = 50
    _mask  = np.zeros((1024,1024,3), np.float32)
    for region in regions:
        y1,y2,x1,x2 = region
        y1 = y1 * 2
        y2 = y2 * 2
        x1 = x1 * 2
        x2 = x2 * 2
        """
        _mask[y2:y2+pad,x1:x2,:]=1
        _mask[y1-pad:y1,x1:x2,:]=1
        _mask[y1:y2,x1 - pad :x1,:]=1
        _mask[y1:y2,x2:x2+pad,:]=1
        """
        _mask[y1:y2,x1:x2,:]=1

    return torch.from_numpy(_mask).unsqueeze(0).permute((0,3,1,2)).float()

def getBack(var_grad_fn):
    """get backpropagate gradient.
    """
    logger.info(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                logger.info(n[0])
                logger.info(f'Tensor with grad found: {tensor}')
                logger.info(f' - gradient: {tensor.grad}\n')
            except AttributeError as e:
                getBack(n[0])

def get_face_info(
                  image: np.ndarray,
                  detector: object
                 ):
    """get face info function.
    """
    res = infer(detector, image)
    ldm = get_landmarks_from_mediapipe_results(res)
    h,w = image.shape[:2]
    ldm[:,0] *= w # 1024x1024
    ldm[:,1] *= h # 1024x1024
    angle = get_euler_angle(res)   
    pitch, yaw, _ = angle
    is_eye_closed = need_to_warp(res)

    return edict(
                   landmarks = ldm,
                   is_eye_closed = is_eye_closed,
                   pitch = pitch,
                   yaw = yaw
                )

def gen_masks(ldm, image):
    """gen masks function
    """
    masks = {}
    landmarks_dict = {}

    # oval
    landmarks_dict["oval"] = [[(10, 338),(338, 297),(297, 332),(332, 284),(284, 251),(251, 389),(389, 356),(356, 454),(454, 323),(323, 361),(361, 288),(288, 397),(397, 365),(365, 379),(379, 378),(378, 400),(400, 377),(377, 152),(152, 148),(148, 176),(176, 149),(149, 150),(150, 136),(136, 172),(172, 58),(58, 132),(132, 93),(93, 234),(234, 127),(127, 162),(162, 21),(21, 54),(54, 103),(103, 67),(67, 109),(109, 10)]]

    # left eye
    landmarks_dict["eyes"] = [[(263, 249),(249, 390),(390, 373),(373, 374),(374, 380),(380, 381),(381, 382),(382, 362),(263, 466),(466, 388),(388, 387),(387, 386),(386, 385),(385, 384),(384, 398),(398, 362), (362, 263)], [(33, 7),(7, 163),(163, 144),(144, 145),(145, 153),(153, 154),(154, 155),(155, 133),(33, 246),(246, 161),(161, 160),(160, 159),(159, 158),(158, 157),(157, 173),(173, 133)]]

    # left eyebrow
    landmarks_dict["eyebrow"] = [[(276, 283),(283, 282),(282, 295),(295, 285),(300, 293),(293, 334),(334, 296),(296, 336)],  [(46, 53),(53, 52),(52, 65),(65, 55),(70, 63),(63, 105),(105, 66),(66, 107)]]

    # left eye iris
    landmarks_dict["gaze"] = [[(474, 475),(475, 476),(476, 477),(477, 474)], [(469, 470),(470, 471),(471, 472),(472, 469)]]
    
    # lips
    landmarks_dict["lips"] = [[(61, 146),(146, 91),(91, 181),(181, 84),(84, 17),(17, 314),(314, 405),(405, 321),(321, 375),(375, 291),(61, 185),(185, 40),(40, 39),(39, 37),(37, 0),(0, 267),(267, 269),(269, 270),(270, 409),(409, 291),(78, 95),(95, 88),(88, 178),(178, 87),(87, 14),(14, 317),(317, 402),(402, 318),(318, 324),(324, 308),(78, 191),(191, 80),(80, 81),(81, 82),(82, 13),(13, 312),(312, 311),(311, 310),(310, 415),(415, 308)]]

    for k, vs in landmarks_dict.items():
        mask = np.zeros_like(image).astype(np.uint8)
        for v in vs:
            points_tmp = []
            for x in v:
                points_tmp += [ldm[x[0], :], ldm[x[1], :]]
            points = np.array(points_tmp).astype(np.int32)
            mask = cv2.fillPoly(mask.copy(), np.int32([points]), (1,1,1))
        masks[k] = torch.from_numpy(mask).unsqueeze(0).permute((0,3,1,2)).cuda().float()

    mask_like_chin = torch.zeros_like(masks["oval"])
    h,w = mask_like_chin.shape[2:]
    mask_like_chin[:, :, h//2:h, :] = 1
    masks["chin"] =  masks["oval"] * mask_like_chin #torch.from_numpy(chin_mask).unsqueeze(0).permute((0,3,1,2)).cuda().float()
    return masks

def get_latents(
                path: str,
                myself_e4e_path: str = None,
                w_path=None,
               ):
    e4e = Encoder4EditingWrapper(e4e_path if myself_e4e_path is None else myself_e4e_path)
    files = [os.path.join(path, x) for x in os.listdir(path)]
    files = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))
    for i, _path in enumerate(files):
        image = np.float32(cv2.imread(_path) / 255.0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        image_tensor = 2 * (to_tensor(image).to("cuda") - 0.5)
        with torch.no_grad():
            latent = e4e(image_tensor)
        torch.save(latent.cpu().detach(), os.path.join(w_path, f'{i + 1}.pt'))

def select_id_latent(
                     latent_path: str,
                     image_path: str,
                     gen_path: str,
                     Decoder: object,
                     device: str,
                    ):
    from .loss.id_loss import IDLoss
    files = [os.path.join(latent_path, x) for x in os.listdir(latent_path)]
    files = sorted(files, key = lambda x: int(os.path.basename(x).split('.')[0]))

    image_files = [os.path.join(image_path, x) for x in os.listdir(image_path)]
    image_files = sorted(image_files, key = lambda x: int(os.path.basename(x).split('.')[0]))
    metric = IDLoss()
    _metric_value = torch.tensor([999.0], dtype = torch.float32).to(device)
    Decoder = Decoder.to(device)
    selected_id = 0
    selected_id_latent = None
    selected_id_image = None
    gen_files_list = []
    for i, _path in enumerate(files):
        #_image_file_path = os.path.basename(_path)
        _image_file_path = image_files[i]
        image = np.float32(cv2.imread(_image_file_path) / 255.0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256,256), interpolation = cv2.INTER_CUBIC)
        image_tensor = 2 * (to_tensor(image).to("cuda") - 0.5)
        latent = torch.load(_path).to(device)
        with torch.no_grad():
            gen_tensor = Decoder(latent)
        value = metric(gen_tensor, image_tensor)
        if value < _metric_value:
            _metric_value = value
            selected_id = i
            selected_id_latent = latent
            selected_id_image = gen_tensor

        gen_tensor = gen_tensor * 0.5 + 0.5
        image_gen = from_tensor(gen_tensor) * 255.0
        image_gen = cv2.cvtColor(image_gen, cv2.COLOR_RGB2BGR)
        gen_file_path = os.path.join(gen_path, f"{i}_gen.jpg")
        gen_files_list.append(gen_file_path)
        cv2.imwrite(gen_file_path, image_gen)
        
    return gen_files_list,\
           cv2.cvtColor(from_tensor(selected_id_image * 0.5 + 0.5) * 255.0, cv2.COLOR_RGB2BGR), \
           selected_id_latent,\
           selected_id

def pose_optimization(
                      latent_id: torch.Tensor,
                      id_image: np.ndarray,
                      ground_truth: np.ndarray,
                      res_gt: edict,
                      res_id: edict,
                      G: Callable,
                      pose_edit: Callable,
                      loss_register: Callable,
                      **kwargs
                     ):

    torch.cuda.empty_cache()
    if DEBUG:
        from torchvision.utils import make_grid
    
    if VERBOSE:
        t = Timer()
    target_res = kwargs.get("target_res", 1024)

    h,w = ground_truth.shape[:2]
    if h != target_res:
        ground_truth = cv2.resize(ground_truth, (target_res, target_res))
    #mask_gt = face_parse(ground_truth)
    #erosion_size = 15
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    #mask_gt = cv2.erode(mask_gt, element)

    #if mask_gt.ndim < 3:
    #    mask_gt = mask_gt[..., np.newaxis]

    device = kwargs.get("device", "cuda")
    G = G.to(device)

    id_image = np.float32(id_image / 255.0)
    ground_truth = np.float32(ground_truth / 255.0)

    mask_gt = np.zeros((h,w,1))
    landmarks_gt = np.int32(res_gt.landmarks)
    points_gt = np.array([landmarks_gt[x[0],:] for x in points]).astype(np.int32)
    mask_gt = cv2.fillPoly(mask_gt, np.int32([points_gt]), (1,1,1))

    mask_id = np.zeros((h,w,1))
    landmarks_id = np.int32(res_id.landmarks)
    points_id = np.array([landmarks_id[x[0],:] for x in points]).astype(np.int32)
    mask_id = cv2.fillPoly(mask_id, np.int32([points_id]), (1,1,1))

    # add mask to paste
    mask_to_paste, _ = get_mask_by_region()
    mask_to_paste_resize = cv2.resize(mask_to_paste, (target_res,target_res))
    mask_to_paste_resize_tensor = to_tensor(mask_to_paste_resize).to(device)
    
    mask_gt_region = mask_gt

    #mask_gt_region = np.ones_like(mask_gt) #np.int32((mask_to_paste_resize + mask_gt) >= 1)
    #mask_gt_region = np.int32((mask_to_paste_resize + mask_gt) >= 1)

    scale = 1024 // target_res

    pad = 50 // scale
    mask_facial = np.ones((target_res,target_res,1), dtype = np.float32)
    pad_x = pad - 10 // scale
    pad_mouth = pad - 20 // scale
    mask_facial[310 // scale + pad: 556// scale - pad, 258 // scale + pad_x: 484 // scale - pad_x] = 0
    mask_facial[310 // scale + pad:558 // scale - pad, 536 // scale + pad_x: 764 // scale - pad_x] = 0
    mask_facial[620 // scale + pad:908 // scale - pad, 368 // scale + pad_mouth: 656 // scale - pad_mouth] = 0
        
    mask_gt_tensor = to_tensor(mask_gt_region).to(device)
    mask_id_tensor = to_tensor(mask_id).to(device)
    mask_facial_tensor = to_tensor(mask_facial).to(device)

    #mask_gt_tensor = 2 * (mask_gt_tensor - 0.5)
    #mask_facial_tensor = 2 * (mask_facial_tensor - 0.5)


    gt_tensor = to_tensor(ground_truth).to(device)
    gt_tensor = 2 * (gt_tensor -  0.5)

    gt_tensor.requires_grad = False
    mask_gt_tensor.requires_grad = False

    epochs = kwargs.get("epochs", 10)
    with torch.no_grad():
        yaw, pitch = res_id.yaw, res_id.pitch
        id_zflow = pose_edit(latent_id, yaw, pitch)

    batchsize = kwargs.get("batchsize", 1)

    resume_param = kwargs.get("resume_param", None)
    if resume_param is None:
        yaw_to_optim = torch.tensor([0.0 for _ in range(batchsize)]).type(torch.FloatTensor).to(device)#torch.from_numpy(np.array([0.0])).type(torch.FloatTensor).to(device)
        pitch_to_optim = torch.tensor([0.0 for _ in range(batchsize)]).type(torch.FloatTensor).to(device)#torch.from_numpy(np.array([0.0])).type(torch.FloatTensor).to(device)
    else:
        yaw_to_optim, pitch_to_optim = resume_param
        yaw_to_optim = yaw_to_optim.detach().reshape(1).to(device)
        pitch_to_optim = pitch_to_optim.detach().reshape(1).to(device)

    yaw_to_optim.requires_grad = True
    pitch_to_optim.requires_grad = True

    #optim = torch.optim.LBFGS([yaw_to_optim, pitch_to_optim], lr = kwargs.get("lr", 1.0), max_iter = 20)
    optim = torch.optim.SGD([yaw_to_optim, pitch_to_optim], lr = kwargs.get("lr", 1.0))
    #optim = torch.optim.Adam([yaw_to_optim, pitch_to_optim], lr = kwargs.get("lr", 1.0))

    sche = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5)
    if VERBOSE:
        t.tic("optimize pose")

    writer = kwargs.get("writer", None)
    # force resize

    #gt_tensor = torch.nn.functional.interpolate(gt_tensor, (256, 256))
    #mask_gt_tensor = torch.nn.functional.interpolate(mask_gt_tensor, (256, 256))
    #mask_id_tensor = torch.nn.functional.interpolate(mask_id_tensor, (256, 256))

    threshold = kwargs.get("threshold", 0.02)

    for epoch in range(1, epochs + 1):
        def closure():
            optim.zero_grad()
            w = pose_edit(id_zflow, 
                          yaw_to_optim,
                          pitch_to_optim,
                          True)
            style_space = G.get_style_space(w)
            gen_tensor = G(style_space)
            #gen_tensor = torch.nn.functional.interpolate(gen_tensor, (256, 256))
            ret = loss_register(gen_tensor * mask_gt_tensor,  gt_tensor * mask_gt_tensor, mask_facial_tensor, is_gradient = False)
            if batchsize != 1:
                gradient_value = torch.Tensor([1. for _ in range(batchsize)]).to(ret['loss'])
                #ret = loss_register(gen_tensor,  gt_tensor, is_gradient = False)
                ret['loss'].backward(gradient = gradient_value, retain_graph = True)
            else:
                ret['loss'].backward(retain_graph = True)
            return ret["loss"]
        

        #if ret['loss'].item() < threshold:
        #    logger.info(f"less {threshold}, stop training....")
        #    break
        optim.step(closure)
        sche.step()

        if writer is not None and DEBUG:
            w = pose_edit(id_zflow, 
                          yaw_to_optim,
                          pitch_to_optim,
                          True)
            style_space = G.get_style_space(w)
            gen_tensor = G(style_space)
            #with torch.no_grad():
            ret = loss_register(gen_tensor * mask_gt_tensor,  gt_tensor * mask_gt_tensor, mask_facial_tensor, is_gradient = False)
            writer.add_scalars(f'pose_estimate/scalar', ret, global_step = epoch)
            writer.add_scalars(f'pose_estimate/pose', dict(yaw = yaw_to_optim, pitch = pitch_to_optim), global_step = epoch)
            #images_in_training = torch.cat(((1 - mask_to_paste_resize_tensor) * gt_tensor + gen_tensor * mask_to_paste_resize_tensor), dim = 2)
            images_in_training = torch.cat(((1 - mask_gt_tensor) * gt_tensor + gen_tensor * mask_gt_tensor, mask_facial_tensor * mask_gt_tensor * gen_tensor, mask_facial_tensor * mask_gt_tensor * gt_tensor), dim =2)
            writer.add_image(f'pose_estimate/image', make_grid(images_in_training.detach(),normalize = True, scale_each=True), epoch)
    if VERBOSE:
        t.toc("optimize pose")
    w = pose_edit(id_zflow, 
                  yaw_to_optim,
                  pitch_to_optim,
                  True)
    style_space = G.get_style_space(w)
    gen_tensor = G(style_space)
    # reset pose edit latent 
    # to avoid the gradient accumulation.
    pose_edit.reset()

    return w, from_tensor(gen_tensor) * 0.5 + 0.5, torch.cat((yaw_to_optim, pitch_to_optim), dim = 0)

# Stage III: facial attribute optimized
def facial_attribute_optimization(    
                                  w_latent: torch.Tensor,
                                  ground_truth: np.ndarray,
                                  ret_gt: edict,
                                  loss_register: Callable,
                                  ss_decoder: object,
                                  gammas: torch.Tensor = None,
                                  images_tensor_last: torch.Tensor = None,
                                  gt_images_tensor_last: torch.Tensor = None,
                                  **kwargs
                                 ):
    if VERBOSE or DEBUG:
        t = Timer()

    if DEBUG:
        from torchvision.utils import make_grid

    device = kwargs.get("device", "cuda")

    alpha_init = [0] * 32
    alpha_tensor = []
    for x in alpha_init:
        alpha_per_tensor = torch.tensor(x).type(torch.FloatTensor).to(device)
        alpha_per_tensor.requires_grad = True
        alpha_tensor.append(alpha_per_tensor)
    dlatents = ss_decoder.get_style_space(w_latent.detach())
    dlatents_gt = [dlatent.clone().repeat(region_num, 1) for dlatent in dlatents]

    def get_masked_dlatent_in_region(tmp_alpha_tensor):
        dlatents_with_masked = [dlatent.clone().repeat(region_num, 1) for dlatent in dlatents]

        for i, ks in enumerate(masks_name):
            for k in ks:
                alpha_current = alphas_split_into_region[k]
                relative_index_current = alphas_relative_index[k]
                for j, (l,c) in enumerate(alpha_current):
                    r_i = relative_index_current[j]
                    latent = dlatents_with_masked[l]
                    latent[i, c] = latent[i, c] + tmp_alpha_tensor[r_i]
                    dlatents_with_masked[l] = latent
        return dlatents_with_masked

    def update_alpha():
        dlatents_tmp = [dlatent.clone() for dlatent in dlatents]
        count = 0
        # first 5 elements.
        for k, v in alphas:
            for i in v:
                dlatents_tmp[k][:, i] = dlatents[k][:, i] + alpha_tensor[count]
                count += 1
        return dlatents_tmp

    isEyeClosed = ret_gt.is_eye_closed   
    masks_gt = gen_masks(ret_gt.landmarks, ground_truth)
    gt_image_tensor = to_tensor(ground_truth).to(device)
    gt_image_tensor = 2 * (gt_image_tensor -  0.5)

    mask_oval_gt = masks_gt["oval"]
    masks_oval_gt = mask_oval_gt.repeat(region_num, 1, 1, 1)
    gt_images_tensor = gt_image_tensor.repeat(region_num, 1, 1, 1)

    tensors_to_optim = [dict(params = alpha_tensor[i], lr = gammas[i].item()) for i in range(32)]
    optim = torch.optim.AdamW(tensors_to_optim, amsgrad=True)
    sche = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.8)

    weights_all = torch.ones(region_num).to(dlatents[0])
    if isEyeClosed:
        weights_all[-1] = 0.0
    weights = torch.ones(region_num).to(dlatents[0])
    weights[-1] = 0.0

    weights_all.requires_grad = False
    weights.requires_grad = False
    masks_oval_gt.requires_grad = False
    epochs = kwargs.get("epochs", 30)
    writer = kwargs.get("writer", None)

    if VERBOSE:
        t.tic("facial optimize..")

    for epoch in range(1, epochs + 1):
        if DEBUG:
            t.tic("one epoch")
            t.tic("masked dlatent")
        dlatents_gen = get_masked_dlatent_in_region(alpha_tensor)
        if DEBUG:
            t.toc("masked dlatent")
            t.tic("ss decoder")
        #images_tensor = ss_decoder(dlatents_tmp)
        #images_tensor = torch.nn.functional.interpolate(images_tensor, (256, 256))
        if DEBUG:
            t.toc("ss decoder")
            t.tic("loss")
        ret = loss_register(
                             dlatents_gen,
                             dlatents_gt,
                             is_gradient = False
                           )
        if DEBUG:
            t.toc("loss")
            t.tic("optim")
        loss = ret["loss"]
        optim.zero_grad()
        gradient_value = torch.Tensor([1.] * region_num).to(loss)
        loss.backward(gradient = gradient_value,retain_graph = True)
        optim.step()
        if DEBUG:
            t.toc("optim")
            t.tic("sche")
        sche.step()
        if DEBUG:
            t.toc("sche")
            t.toc("one epoch")
        if DEBUG and epoch % 10 == 0:
            #string_to_info = reduce(lambda x, y: x + ', ' + y , [f'{k} {v.mean().item()}' for k, v in ret.items()])
            #logger.debug(f"{epoch}/{epochs}: ... {string_to_info}")
            if writer is not None:
                index = kwargs.get("index", 0)
                loss_to_show = dict()
                for k, v in ret.items():
                    if len(v.shape) >= 1:
                        for i in range(region_num):
                            loss_to_show[k + '_' + region_name[i]] = v[i]
                    else:
                        loss_to_show[k] = v
                writer.add_scalars(
                                    f'loss_{index}', 
                                    loss_to_show,
                                    global_step = epoch,
                                  )
                images_in_training = torch.cat((images_tensor, gt_images_tensor), dim = 2)
                writer.add_image(f'image_training_{index}', make_grid(images_in_training.detach(),normalize=True, scale_each=True), epoch)
                with torch.no_grad():
                    dlatents_all = update_alpha()
                    image_after_update_all = ss_decoder(dlatents_all)
                image_to_show = torch.cat((image_after_update_all, gt_image_tensor),dim = 2)
                writer.add_image(f'image_{index}', make_grid(image_to_show.detach(),normalize=True, scale_each=True), epoch)

    if VERBOSE:
        t.toc("facial optimize..")
    with torch.no_grad():
        dlatents_all = update_alpha()
        image_tensor_after_update_all = ss_decoder(dlatents_all) 
    image_gen = (from_tensor(image_tensor_after_update_all) * 0.5 + 0.5) * 255.0
    return dlatents_all,\
           image_gen, \
           alpha_tensor

def get_facial_gamma(
                     dlatents: List,
                     masks: List,
                     ss_decoder: object,
                     loss: object,
                     device: str
                    ):
    loss.to(device)
    def get_gamma(
                  alpha_tensor_next, 
                  alpha_tensor_pre,
                  gammas
                 ):

        """
        mask = masks_current[region_name]
        alpha_in_region = alphas_split_into_region[region_name]
        alpha_relative_index = alphas_relative_index[region_name]
        """
        masks_gamma = torch.cat([masks[name] for name in region_names], 0)
        dlatents_masked = get_masked_dlatent(alpha_tensor_next, dlatents)
        dlatents_pre_masked = get_masked_dlatent(alpha_tensor_pre, dlatents)

        with torch.no_grad():
            St = ss_decoder(dlatents_masked)
            St1 = ss_decoder(dlatents_pre_masked)
            diff = (alpha_tensor_next - alpha_tensor_pre)
            diff[diff == 0.0] = 1.0
            current_gammas = loss(St1 * masks_gamma, St * masks_gamma, is_reduce = False) / (diff) # 32

        return gammas + current_gammas.detach()
   
    gammas = torch.zeros(32).to(device)
    perturbations = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    for k, p in enumerate(perturbations):
        sigma = torch.tensor(p).repeat(32).to(device)
        if k > 0:
            gammas = get_gamma(
                               sigma,
                               sigma_copy,
                               gammas
                              )
        sigma_copy = sigma.clone()
    gammas = gammas / len(perturbations)
    for region in region_names:
        gammas[alphas_relative_index[region]] = torch.exp(-1.5 * gammas[alphas_relative_index[region]] / gammas[alphas_relative_index[region]].max())

    return gammas

def pipeline_init(
                   config_path: str,
                   save_path: str,
                   path: str = None,
                 ):
    
    with open(os.path.join(config_path, "config.yaml")) as f:
        basis_config = edict(yaml.load(f, Loader=yaml.CLoader))

    if path is None:
        video_path = basis_config.path
    else:
        video_path = path

    if os.path.isdir(video_path):
        face_folder_path = video_path
    else:
        face_folder_path = os.path.join(save_path, "data")
        if not os.path.exists(face_folder_path):
            crop(video_path, face_folder_path)
        else:
            logger.info("re-used last processed data.")
        face_folder_path = os.path.join(face_folder_path, "smooth")

    assert len(os.listdir(face_folder_path)) > 1, "face files not exists."
    
    attachment_info = make_train_dirs(save_path)
    stage_four_path = attachment_info["stage_four_path"]

    finetuning_w_space = basis_config.finetuning_w_space if hasattr(basis_config, "finetuning_w_space") else True
    attachment_info["finetuning_w_space"] = finetuning_w_space
    attachment_info["face_folder_path"] = face_folder_path

    w_path = attachment_info["w_path"]
    if len(os.listdir(face_folder_path)) == len(os.listdir(w_path)):
        logger.info("w space files exits. w space generator skip.")
    else:
        get_latents(
                    face_folder_path,
                    myself_e4e_path=None,
                    w_path=w_path
                   )
    w_snapshots = os.path.join(stage_four_path, "w_snapshots")
    os.makedirs(w_snapshots, exist_ok=True)

    return attachment_info

def pivot_finetuning(
                     path_images: str,
                     path_style_latents: str,
                     path_snapshots: str,
                     ss_decoder: object,
                     config: edict,
                     **kwargs
                    ):
    w_pivot_finetuning = kwargs.get("w_pivot_finetuning", False)
    from torchvision import transforms
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader
    if w_pivot_finetuning:
        from .ImagesDataset import ImagesDataset_W as ImagesDataset
    else:
        from .ImagesDataset import ImagesDataset

    resolution = kwargs.get("resolution", 1024)
    batchsize = kwargs.get("batchsize", 1)
    lr = kwargs.get("lr", 3e-4)
    resume_path = kwargs.get("resume_path", None)
    rank = kwargs.get("rank", -1)
    world_size = kwargs.get("world_size", 0)
    device = "cuda:0"

    if rank != -1:
        device = rank
        dist.init_process_group("nccl", rank=rank, world_size=world_size) 
        torch.cuda.set_device(rank)

    expressive_path = config.expressive_path if hasattr(config, "expressive_path") else None
    ss_path = config.ss_path if hasattr(config, "ss_path") else None
    space_finetuning = config.space_finetuning if hasattr(config, "space_finetuning") else "style_space"
    kmeans_info = config.kmeans_info if hasattr(config, "kmeans_info") else None

    def get_dataloader(
                      ):
    
        if space_finetuning == "style_space":

            if expressive_path is None:
                dataset = ImagesDataset(path_images, path_style_latents, transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    transforms.Resize(size = (resolution, resolution))]),
                    kmeans_info = kmeans_info
                    )
            else:
                dataset = ImagesDatasetV2(path_images, path_style_latents, expressive_path, transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    transforms.Resize(size = (resolution, resolution))]))
        elif space_finetuning == "w_space":
                dataset = ImagesDatasetW(path_images, path_style_latents, transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    transforms.Resize(size = (resolution, resolution))]))
        else:
            raise RuntimeError(f"{space_finetuning} not expected type.")

        if rank != -1:
            batch_size = batchsize // world_size
            return DataLoader(
                              dataset, batch_size = batch_size, \
                              num_workers = min(batchsize, 8),  \
                              #num_workers = 1,  \
                              sampler = DistributedSampler(dataset, shuffle = False, rank = rank, num_replicas = world_size, drop_last = True), \
                              pin_memory=True
                             )
        else:
            return DataLoader(
                              dataset, batch_size = batchsize, \
                              shuffle = False, \
                              num_workers = min(batchsize, 8), drop_last = True
                             )
    
    class PivotLossRegister(LossRegisterBase):
        
        def forward(self, 
                    x,
                    y,
                    mask = None,
                    y_random =None
                   ):
            l2 = self.l2(x,y).mean() * self.l2_weight
            lpips = self.lpips(x,y).mean() * self.lpips_weight
        
            #if y_random is not None:
            if mask is not None:
                #x = x * (1 - mask)
                #y = y * (1 - mask)
                l2_with_mask = self.l2(x, y).mean() * self.l2_mask_weight
                lpips_with_mask = self.lpips(x, y).mean() * self.lpips_mask_weight
    
                return {
                        "l2": l2,
                        "lpips": lpips,
                        "l2_with_mask": l2_with_mask,
                        "lpips_with_mask": lpips_with_mask
                       }
            return {
                    "l2": l2,
                    "lpips": lpips
                   }

    loss_register = PivotLossRegister(config) 

    loss_register.lpips.set_device(device)
    dataloader = get_dataloader()

    for p in ss_decoder.parameters():
        p.requires_grad = True
    
    #parameters = []
    #for k, v in ss_decoder.named_parameters():
    #    if "affine" in k:
    #        logger.info(f"{k} add into optimization list.")
    #        v.requires_grad = True
    #        parameters.append(dict(params = v, lr = lr))
    ss_decoder.to(device)
    optim = torch.optim.Adam(ss_decoder.parameters(), lr = lr)

    lastest_model_path = None
    start_idx = 1
    if resume_path is not None:

        epoch_from_resume = int(''.join(re.findall('[0-9]+', os.path.basename(resume_path))))
        ss_decoder.load_state_dict(torch.load(resume_path))
        logger.info(f"resume from {epoch_from_resume}...")
        start_idx = epoch_from_resume + 1
        total_idx = epoch_from_resume * len(dataloader)

    if rank != -1:
        ss_decoder = DDP(ss_decoder, device_ids = [rank], find_unused_parameters=True)

    #optim = torch.optim.Adam(parameters, lr = lr)
    total_idx = 0
    epochs = kwargs.get("epochs", 100)
    tensorboard = kwargs.get("tensorboard", None)
    writer = None
    if tensorboard is not None and (rank == 0 or rank == -1):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(tensorboard)

    save_interval = kwargs.get("save_interval", 100)
    epoch_pbar = tqdm(range(start_idx, epochs + 1))

    #mask = get_up_bottom_mask()
    #mask = mask.to(device)
    mask_to_paste, mask_boundary = get_mask_by_region()
    mask_to_paste_resize = cv2.resize(mask_to_paste, (resolution,resolution))
    mask = to_tensor(mask_to_paste_resize).to(device)
    mask_boundary = to_tensor(mask_boundary).to(device)

    #mask_dilate_to_paste_resize = cv2.resize(mask_dilate_to_paste, (resolution,resolution))
    #mask = to_tensor(mask_dilate_to_paste_resize).to("cuda")

    min_loss = 0xffff # max value.
    internal_size =  len(dataloader) // 5
    if internal_size <= 0:
        internal_size = 1
    for epoch in epoch_pbar:
        if rank == 0 or rank == -1:
            logger.info(f"internal_size is {internal_size}.")
            epoch_pbar.update(1)
        sample_loss = 0
        sample_count = 0
        for idx, (image, pivot) in enumerate(dataloader):
        
            b,c,h,w = image.shape
            if space_finetuning == "w_space":
                pivot = pivot.to(device)
            elif space_finetuning == "style_space":
                pivot = [x.to(device) for x in pivot]

            image = image.to(device)  
            image_gen = ss_decoder(pivot)

            ret = loss_register(image, image_gen, is_gradient = False)
            loss = ret['loss']

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_idx += 1
            if idx % internal_size == 0 and (rank == 0 or rank == -1):
                sample_loss += loss.mean()
                sample_count += 1
                string_to_info = reduce(lambda x, y: x + ', ' + y , [f'{k} {v.mean().item()}' for k, v in ret.items()])
                logger.info(f"{idx+1}/{epoch}/{epochs}: {string_to_info}")

                if writer is not None:
                    image_to_show = torch.cat((image_gen, image),dim = 2)
                    writer.add_image('image', make_grid(image_to_show.detach(),normalize=True, scale_each=True), total_idx)
                    writer.add_scalars('loss', ret, total_idx)

        if (rank == 0 or rank == -1):
            sample_loss /= sample_count
            if sample_loss < min_loss:
                lastest_model_path = os.path.join(path_snapshots, f"{epoch}.pth")
                torch.save(ss_decoder.state_dict() if rank == -1 else ss_decoder.module.state_dict(), lastest_model_path)
                min_loss = sample_loss
                logger.info(f"min_loss: {min_loss}, epoch {epoch}")

    if rank == 0 or rank == -1:
        import shutil
        shutil.copyfile(lastest_model_path, os.path.join(os.path.dirname(lastest_model_path), "best.pth"))
        logger.info(f"training finished; the lastet snapshot saved in {lastest_model_path}")
        writer.close()
        
    return lastest_model_path

def expressive_encoding_pipeline(
                                 config_path: str,
                                 save_path: str,
                                 path: str = None,
                                 decoder_path: str = None
                                ):
    #TODO: log generator.
    from copy import deepcopy
    G = load_model(stylegan_path).synthesis
    for p in G.parameters():
        p.requires_grad = False

    device = "cuda:0"

    pose_edit = PoseEdit()
    if decoder_path is not None:
        ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(G), to_resolution = 512)
        target_res = 512
    else:
        ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(G))
        target_res = 1024

    ss_decoder.to(device)

    if decoder_path is not None:
        ss_decoder.load_state_dict(torch.load(decoder_path), False)
        logger.info("decoder loading ....")

    for p in ss_decoder.parameters():
        p.requires_grad = False

    attachment_info = pipeline_init(config_path, save_path, path)
    
    if not os.path.exists(attachment_info["cache_path"]):
        gen_file_list, \
        selected_id_image,\
        selected_id_latent, \
        selected_id \
        = select_id_latent(\
                            attachment_info["w_path"],\
                            attachment_info["face_folder_path"],\
                            attachment_info["stage_one_path"], \
                            ss_decoder,\
                            device
                          )
        torch.save([
                    gen_file_list, 
                    seleceted_id_image,
                    selected_id_latent, 
                    selected_id 
                   ],
                   attachment_info["cache_path"]
                  )
    else:
        gen_file_list, \
        selected_id_image,\
        selected_id_latent, \
        selected_id = \
            torch.load(attachment_info["cache_path"])

    detector = get_detector()
    face_info_from_id = get_face_info(
                                      np.uint8(selected_id_image),
                                      detector
                                     )
    masks = gen_masks(face_info_from_id.landmarks, selected_id_image)
    gammas = get_facial_gamma(
                              ss_decoder.get_style_space(selected_id_latent.to(device)),
                              masks,
                              ss_decoder,
                              LPIPS(),
                              device
                             )

    with open(os.path.join(config_path, "pose.yaml")) as f1, \
         open(os.path.join(config_path, "facial_attribute.yaml")) as f2, \
         open(os.path.join(config_path, "pti.yaml")) as f3:
        config_pose = edict(yaml.load(f1, Loader = yaml.CLoader))
        config_facial = edict(yaml.load(f2, Loader = yaml.CLoader))
        config_pti = edict(yaml.load(f3, Loader = yaml.CLoader))

    class PoseLossRegister(LossRegisterBase):

        def forward(self, x, y, mask):
            x = x * mask
            y = y * mask
            l2_loss = self.l2_loss(x, y).mean()
            #l2_loss = self.l2_loss(x, y).mean()
            lpips_loss = self.lpips_loss(x,y).mean()
    
            return {
                     "l2_loss": l2_loss,
                     "lpips_loss": lpips_loss
                   }

    class FacialLossRegister(LossRegisterBase):

        def forward(self,
                    x,
                    y,
                   ):
            n = x[0].shape[0]
            l1_loss = torch.zeros((n)).to(x[0])
            for (_x, _y) in zip(x,y):
                l1_loss += self.l1_loss(_x, _y)
            ret = {
                    "l1_loss": l1_loss
                  }

            return ret

    stage_two_path = attachment_info["stage_two_path"]
    stage_two_param_path = attachment_info["stage_two_param_path"]
    stage_three_path = attachment_info["stage_three_path"]
    expressive_param_path = attachment_info["expressive_param_path"]
    stage_four_path = attachment_info["stage_four_path"]
    face_folder_path = attachment_info["face_folder_path"]
    gen_length = min(len(gen_file_list), 20000) if not DEBUG else 100
    gen_file_list = gen_file_list[:gen_length]
    pose_loss_register = PoseLossRegister(config_pose)
    facial_loss_register = FacialLossRegister(config_facial)

    writer = None
    if DEBUG or VERBOSE:
        from tensorboardX import SummaryWriter
        tensorboard_path = os.path.join(save_path, "tensorboard")
        writer = SummaryWriter(tensorboard_path)
    optimized_latents = list(filter(lambda x: x.endswith('pt'), os.listdir(stage_two_path)))
    start_index = len(optimized_latents) / 2 - 1
    end_index = gen_length

    for ii, _file in enumerate(gen_file_list):

        if ii > end_index or ii < start_index:
            continue

        if ii >= len(gen_file_list):
            logger.info("all file optimized, skip to pti.")
            break
        gen_image = cv2.imread(_file)
        assert gen_image is not None, "file not exits, please check."
        gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)

        # stage 1.5 get face info
        face_info_from_gen = get_face_info(gen_image, detector)

        #stage 2.
        try:
            w_with_pose = torch.load(os.path.join(stage_two_path, f"{ii+1}.pt"),map_location=f'cpu')
            pose_param = torch.load(os.path.join(stage_two_param_path, f"pose_{ii+1}.pt"),map_location=f'cpu')
            w_with_pose = w_with_pose.to(device)
        except:
            w_with_pose, image_posed, pose_param = pose_optimization(
                                                                     selected_id_latent.detach(),
                                                                     np.uint8(selected_id_image),
                                                                     gen_image,
                                                                     face_info_from_gen,
                                                                     face_info_from_id,
                                                                     ss_decoder,
                                                                     pose_edit,
                                                                     pose_loss_register
                                                                    )
            torch.save(w_with_pose, os.path.join(stage_two_path, f"{ii + 1}.pt"))
            torch.save(pose_param, os.path.join(stage_two_param_path, f"pose_{ii + 1}.pt"))

        style_space_latent, image_gen, facial_param = \
            facial_attribute_optimization(\
                                          w_with_pose, \
                                          gen_image, \
                                          face_info_from_gen, \
                                          facial_loss_register, \
                                          ss_decoder, \
                                          gammas \
                                         )

        torch.save([x.detach().cpu() for x in style_space_latent], os.path.join(stage_three_path, f"{ii+1}.pt"))
        torch.save([pose_param, facial_param], os.path.join(expressive_param_path, f"attribute_{ii+1}.pt"))

    # stage 4.
    snapshots = os.path.join(stage_four_path, "snapshots")
    os.makedirs(snapshots, exist_ok = True)
    snapshot_files = os.listdir(snapshots)

    epochs = 100
    pti_or_not = True
    resume_path = None
    if os.path.exists(snapshots) and len(snapshot_files):
        if os.path.exists(os.path.join(snapshots, "best.pth")):
            pti_or_not = False
            latest_decoder_path = os.path.join(snapshots, "best.pth")
        else:
            snapshot_paths = sorted(snapshot_files, key = lambda x: int(x.split('.')[0]))
            latest_decoder_path = os.path.join(snapshots, snapshot_paths[-1])
            epoch_latest = int(''.join(re.findall('[0-9]+' ,snapshot_paths[-1])))
            pti_or_not = False
            if epoch_latest < epochs:
                pti_or_not = True
                resume_path = latest_decoder_path

    if pti_or_not:
        os.makedirs(snapshots, exist_ok = True)
        latest_decoder_path = pivot_finetuning(
                                               face_folder_path, \
                                               stage_three_path, \
                                               snapshots, \
                                               ss_decoder, \
                                               config_pti, \
                                               writer = writer, \
                                               resume_path = resume_path,\
                                               resolution = target_res, \
                                               epochs = 10,
                                               batchsize = 8
                                              )
        if writer is not None:
            writer.close()
        logger.info(f"latest model path is {latest_decoder_path}")
    #latest_decoder_path = './results/pivot_001/snapshots/100.pth'
    #validate_video_path = os.path.join(save_path, "validate_video.mp4")
    #validate_video_gen(
    #                    validate_video_path,
    #                    latest_decoder_path,
    #                    stage_three_path,
    #                    ss_decoder,
    #                    len(gen_file_list),
    #                    face_folder_path
    #                  )
    #logger.info(f"validate video located in {validate_video_path}")

