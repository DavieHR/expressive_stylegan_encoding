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

import numpy as np

from tqdm import tqdm
from easydict import EasyDict as edict
from DeepLog import logger, Timer
from DeepLog.logger import logging
from torchvision import transforms
from PIL import Image
current_dir = os.getcwd()

from .pose_edit_with_flow import PoseEdit
from .decoder import StyleSpaceDecoder, load_model
from .encoder import Encoder4EditingWrapper
from .FaceToolsBox.alignment import get_detector, infer, \
                             get_euler_angle, get_landmarks_from_mediapipe_results, need_to_warp
from .FaceToolsBox.crop_image import crop

from .loss import LossRegisterBase

from .loss.FaceParsing.model import BiSeNet


from .utils import to_tensor, from_tensor
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
stylegan_path = os.path.join(WHERE_AM_I, "third_party/models/stylegan2_ffhq.pkl")
e4e_path = os.path.join(WHERE_AM_I, "third_party/models/e4e_ffhq_encode.pt")
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

output_copy_region="[[274,494,80,432]]"
soft_mask_region="[[340,494,130,-130],[274,340,130,-130]]"
regions = eval(soft_mask_region)
output_copy_region = eval(output_copy_region)

where_am_i = os.path.dirname(os.path.realpath(__file__))
class face_parsing:
    def __init__(self, path = os.path.join(where_am_i, "third_party", "models", "79999_iter.pth")):

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

face_parse = face_parsing()

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

# STAGE 1: select id latent.
def select_id_latent(
                     paths: dict,
                     G: object,
                     gen_path: str,
                     myself_e4e_path: str = None
                    ):
    from .loss.id_loss import IDLoss
    e4e = Encoder4EditingWrapper(e4e_path if myself_e4e_path is None else myself_e4e_path)

    path = paths["driving_face_path"]
    files = [os.path.join(path, x) for x in os.listdir(path)]
    files = sorted(files, key = lambda x: int(os.path.basename(x).split('.')[0]))

    metric = IDLoss()
    _metric_value = torch.tensor([999.0], dtype = torch.float32).to("cuda")
    selected_id = 0
    selected_id_latent = None
    selected_id_image = None
    gen_files_list = []
    for i, _path in enumerate(files):
        image = np.float32(cv2.imread(_path) / 255.0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256,256), interpolation = cv2.INTER_CUBIC)
        image_tensor = 2 * (to_tensor(image).to("cuda") - 0.5)
        with torch.no_grad():
            latent = e4e(image_tensor)
            gen_tensor = G(latent)
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

# Stage II: pose optimized 
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
    if DEBUG:
        from torchvision.utils import make_grid
    
    if VERBOSE:
        t = Timer()

    h,w = ground_truth.shape[:2]
    if h != 1024:
        ground_truth = cv2.resize(ground_truth, (1024,1024))
    #mask_gt = face_parse(ground_truth)
    #erosion_size = 15
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    #mask_gt = cv2.erode(mask_gt, element)

    #if mask_gt.ndim < 3:
    #    mask_gt = mask_gt[..., np.newaxis]

    id_image = np.float32(id_image / 255.0)
    ground_truth = np.float32(ground_truth / 255.0)

    mask_gt = np.zeros((h,w,1))
    landmarks_gt = np.int32(res_gt.landmarks)
    points_gt = np.array([landmarks_gt[x[0],:] for x in points]).astype(np.int32)
    mask_gt = cv2.fillPoly(mask_gt, np.int32([points_gt]), (1,1,1))

    """
    # find top y
    h, w = mask_gt.shape[:2]

    top_y = 0
    for i in range(h):
        if mask_gt[i, :].sum() > 0:
            top_y = i
            break

    left_x = 0
    # find left x
    for i in range(w):
        if mask_gt[:, i].sum() > 0:
            left_x = i
            break

    # find right x
    right_x = 0
    for i in range(w):
        if mask_gt[:, -i].sum() > 0:
            right_x = -i
            break

    pad_top = 100
    pad_down = 200
    mask_gt[top_y - pad_top: top_y + pad_down, left_x: right_x] = 1
    """

    mask_id = np.zeros((h,w,1))
    landmarks_id = np.int32(res_id.landmarks)
    points_id = np.array([landmarks_id[x[0],:] for x in points]).astype(np.int32)
    mask_id = cv2.fillPoly(mask_id, np.int32([points_id]), (1,1,1))

    # add mask to paste
    mask_to_paste, _ = get_mask_by_region()
    mask_to_paste_resize = cv2.resize(mask_to_paste, (1024,1024))
    mask_to_paste_resize_tensor = to_tensor(mask_to_paste_resize).to("cuda")
    
    mask_gt_region = mask_gt

    #mask_gt_region = np.ones_like(mask_gt) #np.int32((mask_to_paste_resize + mask_gt) >= 1)
    #mask_gt_region = np.int32((mask_to_paste_resize + mask_gt) >= 1)

    pad = 50
    mask_facial = np.ones((1024,1024,1), dtype = np.float32)
    pad_x = pad - 10
    pad_mouth = pad - 20
    mask_facial[310 + pad:556 - pad, 258 + pad_x: 484 - pad_x] = 0
    mask_facial[310 + pad:558 - pad, 536 + pad_x: 764 - pad_x] = 0
    mask_facial[620 + pad:908 - pad, 368 + pad_mouth: 656 - pad_mouth] = 0
        
    mask_gt_tensor = to_tensor(mask_gt_region).to("cuda")
    mask_id_tensor = to_tensor(mask_id).to("cuda")

    mask_facial_tensor = to_tensor(mask_facial).to("cuda")

    gt_tensor = to_tensor(ground_truth).to("cuda")
    gt_tensor = 2 * (gt_tensor -  0.5)
    gt_tensor.requires_grad = False
    mask_gt_tensor.requires_grad = False

    epochs = kwargs.get("epochs", 20)
    yaw, pitch = res_id.yaw, res_id.pitch
    with torch.no_grad():
        id_zflow = pose_edit(latent_id, yaw, pitch)

    resume_param = kwargs.get("resume_param", None)
    if resume_param is None:
        yaw_to_optim = torch.tensor([0.0]).type(torch.FloatTensor).to("cuda")#torch.from_numpy(np.array([0.0])).type(torch.FloatTensor).to("cuda")
        pitch_to_optim = torch.tensor([0.0]).type(torch.FloatTensor).to("cuda")#torch.from_numpy(np.array([0.0])).type(torch.FloatTensor).to("cuda")
    else:
        yaw_to_optim, pitch_to_optim = resume_param
        yaw_to_optim = yaw_to_optim.detach().reshape(1)
        pitch_to_optim = pitch_to_optim.detach().reshape(1)

    yaw_to_optim.requires_grad = True
    pitch_to_optim.requires_grad = True

    optim = torch.optim.Adam([yaw_to_optim, pitch_to_optim], lr = kwargs.get("lr", 1.0))
    sche = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)
    if VERBOSE:
        t.tic("optimize pose")

    writer = kwargs.get("writer", None)
    # force resize

    #gt_tensor = torch.nn.functional.interpolate(gt_tensor, (256, 256))
    #mask_gt_tensor = torch.nn.functional.interpolate(mask_gt_tensor, (256, 256))
    #mask_id_tensor = torch.nn.functional.interpolate(mask_id_tensor, (256, 256))

    threshold = kwargs.get("threshold", 0.02)

    for epoch in range(1, epochs + 1):
        w = pose_edit(id_zflow, 
                      yaw_to_optim,
                      pitch_to_optim,
                      True)
        style_space = G.get_style_space(w)
        gen_tensor = G(style_space)
        #gen_tensor = torch.nn.functional.interpolate(gen_tensor, (256, 256))
        optim.zero_grad()
        ret = loss_register(gen_tensor * mask_gt_tensor,  gt_tensor * mask_gt_tensor, mask_facial_tensor, is_gradient = False)
        #ret = loss_register(gen_tensor,  gt_tensor, is_gradient = False)
        if ret['loss'].item() < threshold:
            logger.info(f"less {threshold}, stop training....")
            break

        ret['loss'].backward(retain_graph = True)

        if writer is not None and DEBUG:
            writer.add_scalars(f'pose_estimate/scalar', ret, global_step = epoch)
            writer.add_scalars(f'pose_estimate/pose', dict(yaw = yaw_to_optim, pitch = pitch_to_optim), global_step = epoch)
            #images_in_training = torch.cat(((1 - mask_to_paste_resize_tensor) * gt_tensor + gen_tensor * mask_to_paste_resize_tensor), dim = 2)
            images_in_training = torch.cat(((1 - mask_gt_tensor) * gt_tensor + gen_tensor * mask_gt_tensor, mask_gt_tensor * gen_tensor, mask_gt_tensor * gt_tensor, mask_facial_tensor * gen_tensor), dim =2)
            writer.add_image(f'pose_estimate/image', make_grid(images_in_training.detach(),normalize=True, scale_each=True), epoch)
        optim.step()
        sche.step()
    if VERBOSE:
        t.toc("optimize pose")
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

    alpha_init = [0] * 32
    alpha_tensor = []
    for x in alpha_init:
        alpha_per_tensor = torch.tensor(x).type(torch.FloatTensor).cuda()
        alpha_per_tensor.requires_grad = True
        alpha_tensor.append(alpha_per_tensor)
    dlatents = ss_decoder.get_style_space(w_latent.detach())
    ground_truth = np.float32(ground_truth / 255.0)
    gt_image_tensor = to_tensor(ground_truth).to("cuda")
    gt_image_tensor = (gt_image_tensor - 0.5) * 2

    if images_tensor_last is not None and gt_images_tensor_last is not None:
        images_tensor_last.requires_grad = False
        gt_images_tensor_last.requires_grad = False
    
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

    # get dlatent with masked
    def get_masked_dlatent(tmp_alpha_tensor):
        dlatents_with_masked = [dlatent.clone().repeat(32, 1) for dlatent in dlatents]
        for index, (k, i) in enumerate(alphas):
            #r_i = alpha_relative_index[index]
            r_i = index
            latents = dlatents_with_masked[k]
            latents[r_i, i] = latents[r_i, i] + tmp_alpha_tensor[r_i]
            dlatents_with_masked[k] = latents
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
    mask_oval_gt = masks_gt["oval"]
    masks_oval_gt = mask_oval_gt.repeat(region_num, 1, 1, 1)
    gt_images_tensor = gt_image_tensor.repeat(region_num, 1, 1, 1)

    def get_gamma(alpha_tensor_next, 
                  alpha_tensor_pre,
                  gammas):

        """
        mask = masks_current[region_name]
        alpha_in_region = alphas_split_into_region[region_name]
        alpha_relative_index = alphas_relative_index[region_name]
        """
        
        masks_gamma = torch.cat([masks_gt[name] for name in region_names], 0)
        dlatents_masked = get_masked_dlatent(alpha_tensor_next)
        dlatents_pre_masked = get_masked_dlatent(alpha_tensor_pre)

        with torch.no_grad():
            St = ss_decoder(dlatents_masked)
            St1 = ss_decoder(dlatents_pre_masked)
            diff = (alpha_tensor_next - alpha_tensor_pre)
            diff[diff == 0.0] = 1.0
            current_gammas = loss_register.lpips(St1 * masks_gamma, St * masks_gamma, is_reduce = False) / (diff) # 32

        return gammas + current_gammas.detach()
   
    if gammas is None:
        gammas = torch.zeros(32).to(alpha_tensor[0])
        perturbations = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        for k, p in enumerate(perturbations):
            sigma = torch.tensor(p).repeat(32).to(alpha_tensor[0])
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
    
    # force resize to small image
    #gt_images_tensor = torch.nn.functional.interpolate(gt_images_tensor, (256, 256))
    #masks_oval_gt = torch.nn.functional.interpolate(masks_oval_gt, (256, 256))
    #if images_tensor_last is not None:
    #    gt_images_tensor_last = torch.nn.functional.interpolate(gt_images_tensor_last, (256, 256))
    #    images_tensor_last = torch.nn.functional.interpolate(images_tensor_last, (256, 256))

    for epoch in range(1, epochs + 1):
        if DEBUG:
            t.tic("one epoch")
            t.tic("masked dlatent")
        dlatents_tmp = get_masked_dlatent_in_region(alpha_tensor)
        if DEBUG:
            t.toc("masked dlatent")
            t.tic("ss decoder")
        images_tensor = ss_decoder(dlatents_tmp)
        #images_tensor = torch.nn.functional.interpolate(images_tensor, (256, 256))
        if DEBUG:
            t.toc("ss decoder")
            t.tic("loss")
        ret = loss_register(
                            images_tensor, 
                            gt_images_tensor,
                            masks_oval_gt,
                            weights_all,
                            weights,
                            is_gradient = False,
                            x_pre = images_tensor_last,
                            y_pre = gt_images_tensor_last
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
           images_tensor.detach(),\
           gt_images_tensor.detach(),\
           gammas, \
           image_gen, \
           alpha_tensor

def pivot_finetuning(
                     path_images: str,
                     path_style_latents: str,
                     path_snapshots: str,
                     ss_decoder: object,
                     config: edict,
                     **kwargs
                    ):
    from torchvision import transforms
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader
    from .ImagesDataset import ImagesDataset, ImagesDatasetV2, ImagesDatasetV3

    resolution = kwargs.get("resolution", 1024)
    batchsize = kwargs.get("batchsize", 1)
    lr = kwargs.get("lr", 3e-4)
    resume_path = kwargs.get("resume_path", None)

    expressive_path = config.expressive_path if hasattr(config, "expressive_path") else None
    ss_path = config.ss_path if hasattr(config, "ss_path") else None

    def get_dataloader(
                      ):

        if expressive_path is None:
            dataset = ImagesDataset(path_images, path_style_latents, transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.Resize(size = (resolution, resolution))]))
        else:
            dataset = ImagesDatasetV2(path_images, path_style_latents, expressive_path, ss_path, transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.Resize(size = (resolution, resolution))]))

        return DataLoader(dataset, batch_size = batchsize, shuffle = False, num_workers = 8)

    class LossRegister(LossRegisterBase):
        def forward(self, 
                    x,
                    y,
                    mask,
                    y_random =None
                   ):
            l2 = self.l2(x,y).mean() * self.l2_weight
            lpips = self.lpips(x,y).mean() * self.lpips_weight
        
            #if y_random is not None:
            if mask is not None:
                x = x * mask #(1 - mask)
                y = y * mask #(1 - mask)
                l2_with_mask = self.l2_mask(x, y).mean() * self.l2_mask_weight
                lpips_with_mask = self.lpips_mask(x, y).mean() * self.lpips_mask_weight

            return {
                    "l2": l2,
                    "lpips": lpips,
                    #"l2_with_mask": l2_with_mask,
                    #"lpips_with_mask": lpips_with_mask
                   }

    loss_register = LossRegister(config) 

    dataloader = get_dataloader()

    for p in ss_decoder.parameters():
        p.requires_grad = True
    device = "cuda:0"
    #parameters = []
    #for k, v in ss_decoder.named_parameters():
    #    if "affine" in k:
    #        logger.info(f"{k} add into optimization list.")
    #        v.requires_grad = True
    #        parameters.append(dict(params = v, lr = lr))

    ss_decoder.to(device)
    optim = torch.optim.Adam(ss_decoder.parameters(), lr = lr)
    #optim = torch.optim.Adam(parameters, lr = lr)
    total_idx = 0
    epochs = kwargs.get("epochs", 100)
    writer = kwargs.get("writer", None)
    lastest_model_path = None
    start_idx = 1
    if resume_path is not None:

        epoch_from_resume = int(''.join(re.findall('[0-9]+', os.path.basename(resume_path))))
        ss_decoder.load_state_dict(torch.load(resume_path))
        logger.info(f"resume from {epoch_from_resume}...")
        start_idx = epoch_from_resume + 1
        total_idx = epoch_from_resume * len(dataloader)

    save_interval = kwargs.get("save_interval", 100)
    epoch_pbar = tqdm(range(start_idx, epochs + 1))

    #mask = get_up_bottom_mask()
    #mask = mask.to(device)
    mask_to_paste, mask_boundary = get_mask_by_region()
    mask_to_paste_resize = cv2.resize(mask_to_paste, (resolution,resolution))
    mask = to_tensor(mask_to_paste_resize).to("cuda")
    mask_boundary = None #to_tensor(mask_boundary).to("cuda")

    #mask_dilate_to_paste_resize = cv2.resize(mask_dilate_to_paste, (resolution,resolution))
    #mask = to_tensor(mask_dilate_to_paste_resize).to("cuda")

    min_loss = 0xffff # max value.
    internal_size = 100 if not DEBUG else 100

    for epoch in epoch_pbar:
        sample_loss = 0
        sample_count = 0
        for idx, (image, pivot) in enumerate(dataloader):

            pivot_random = None
            if not isinstance(pivot, list):
                pivot, pivot_random = [x.to(device) for x in pivot[0]], [y.to(device) for y in pivot[1]]
                #pivot[0] = ss_decoder.get_style_space(pivot[0].to(device))
                #pivot = [(x.to(device), y.to(device)) for (x, y) in zip(pivot[0], pivot[1])]
            else:
                pivot = [x.to(device) for x in pivot]
            image = image.to(device)  


            image_gen = ss_decoder(pivot)

            ret = loss_register(image, image_gen, mask_boundary, is_gradient = False)
            if pivot_random is None:
                ret = loss_register(image, image_gen, mask_boundary, is_gradient = False)
            else:
                image_random = ss_decoder(pivot_random)
                ret = loss_register(image, image_gen, mask_boundary, image_random, is_gradient = False)
            loss = ret['loss']
            optim.zero_grad()
            b = image_gen.shape[0]
            loss.backward()
            optim.step()
            total_idx += 1
            if idx % internal_size == 0:
                sample_loss += loss.mean()
                sample_count += 1
                string_to_info = reduce(lambda x, y: x + ', ' + y , [f'{k} {v.mean().item()}' for k, v in ret.items()])
                logger.info(f"{idx+1}/{epoch}/{epochs}: {string_to_info}")

                #if idx == 0 and writer is not None:
                if writer is not None:
                    image_to_show = torch.cat((image_gen, image, mask * image_gen + (1 - mask) * image),dim = 2)
                    if pivot_random is not None:
                        image_to_show = torch.cat((image_to_show, mask * image_random + (1 - mask) * image, image_random * mask_boundary), dim = 2)

                    writer.add_image(f'image', make_grid(image_to_show.detach(),normalize=True, scale_each=True), total_idx)
                if writer is not None:
                    writer.add_scalars('loss', ret, total_idx)
        if sample_count == 0:
            sample_count += 1
        sample_loss /= sample_count
        if sample_loss < min_loss:
            lastest_model_path = os.path.join(path_snapshots, f"{epoch}.pth")
            torch.save(ss_decoder.state_dict(), os.path.join(path_snapshots, f"{epoch}.pth"))
            min_loss = sample_loss
            logger.info(f"min_loss: {min_loss}, epoch {epoch}")
    return lastest_model_path

def validate_video_gen(
                        save_video_path:str,
                        state_dict_path: str,
                        latents: Union[str, List[np.ndarray]],
                        ss_decoder: Callable,
                        video_length: int,
                        face_folder_path: str,
                        attribute_path: str = None
                      ):

    def update_region_offset(
                              dlatents,
                              offset,
                              region_range
                            ):
        dlatents_tmp = [latent.clone() for latent in dlatents]
        count = 0
        #first 5 elements.
        #forbidden_list = [
        #                  ( 6, 378 ),
        #                  ( 5, 50 ),
        #                  ( 5, 505 )
        #                 ]
        #forbidden_list = []
        #if tuple([k, i]) in forbidden_list:
        #    logger.info(f"{k} {i} is forbidden.")
        #    continue
        for k, v in alphas[region_range[0]:region_range[1]]:
            for i in v:
                dlatents_tmp[k][:, i] = dlatents[k][:, i] + offset[:,count]
                count += 1
        return dlatents_tmp
    if video_length == -1:
        files = list(filter(lambda x: x.endswith('pt'), os.listdir(latent_folder)))
        assert len(files), "latent_folder has no latent file."
        video_length = len(files)
    if state_dict_path is not None:
        ss_decoder.load_state_dict(torch.load(state_dict_path), False)
    with imageio.get_writer(save_video_path, fps = 25) as writer:
        for index in tqdm(range(video_length)):
            if isinstance(latents, str):
                #style_space_latent = torch.load(os.path.join(latents, f"{index+1}.pt"))
                style_space_latent = torch.load(os.path.join(latents, f"{1}.pt"))
                if isinstance(style_space_latent, list):
                    style_space_latent = [s.to("cuda") for s in style_space_latent]
            else:
                style_space_latent = latents[index]

            if not isinstance(style_space_latent, list):
                style_space_latent = style_space_latent.to("cuda:0")
                style_space_latent = ss_decoder.get_style_space(style_space_latent)

            if attribute_path is not None:
                attribute = torch.load(os.path.join(attribute_path, f"{index + 1}.pt"))
                if isinstance(attribute, list) and len(attribute) == 2:
                    attribute = attribute[1]

                attribute = torch.tensor(attribute).reshape(1, -1).to("cuda:0")
                style_space_latent = update_region_offset(style_space_latent, attribute, [5, 8])

            image = np.uint8(np.clip(from_tensor(ss_decoder(style_space_latent) * 0.5 + 0.5), 0.0, 1.0) * 255.0)
            image_gt_path = os.path.join(face_folder_path, f'{index}.png')
            if not os.path.exists(image_gt_path):
                image_gt_path = image_gt_path.replace('png', 'jpg')
            image_gt = cv2.imread(image_gt_path)[...,::-1]
            image_gt = cv2.resize(image_gt, (512,512))
            image_concat = np.concatenate((image, image_gt), axis = 0)
            writer.append_data(image_concat)

            if state_dict_path is None:
                workdir = os.path.join(os.path.dirname(save_video_path), "images")
                os.makedirs(workdir,exist_ok = True)
                cv2.imwrite(os.path.join(workdir, f'{index + 1}.jpg'), image[...,::-1])


def expressive_encoding_pipeline(
                                 config_path: str,
                                 save_path: str,
                                 path: str = None
                                ):

    #TODO: log generator.
    from copy import deepcopy
    G = load_model(stylegan_path).synthesis
    for p in G.parameters():
        p.requires_grad = False

    pose_edit = PoseEdit()
    ss_decoder = StyleSpaceDecoder(synthesis = deepcopy(G))
    for p in ss_decoder.parameters():
        p.requires_grad = False

    with open(os.path.join(config_path, "config.yaml")) as f:
        basis_config = edict(yaml.load(f, Loader = yaml.CLoader))
    
    if path is None:
        video_path = basis_config.path
    else:
        video_path = path

    if os.path.isdir(video_path):
        face_folder_path = video_path
    else:
        face_folder_path = os.path.join(save_path, "data")
        if not os.path.exists(face_folder_path):
            crop(video_path,face_folder_path)
        else:
            logger.info("re-used last processed data.")
        face_folder_path = os.path.join(face_folder_path, "smooth")

    assert len(os.listdir(face_folder_path)) > 1, "face files not exists."

    stage_one_path = os.path.join(save_path, "e4e")  
    stage_two_path = os.path.join(save_path, "pose")  
    stage_three_path = os.path.join(save_path, "facial")  
    stage_four_path = os.path.join(save_path, "pti")  
    expressive_param_path = os.path.join(save_path, "expressive")
    cache_path = os.path.join(save_path, "cache.pt")

    os.makedirs(stage_one_path, exist_ok = True)
    os.makedirs(stage_two_path, exist_ok = True)
    os.makedirs(stage_three_path, exist_ok = True)
    os.makedirs(stage_four_path, exist_ok = True)
    os.makedirs(expressive_param_path, exist_ok = True)

    writer = None
    if DEBUG or VERBOSE:
        from tensorboardX import SummaryWriter
        tensorboard_path = os.path.join(save_path, "tensorboard")
        writer = SummaryWriter(tensorboard_path)

    detector = get_detector()

    # stage 1.

    files_path = {
                    "driving_face_path": face_folder_path
                 }

    if os.path.exists(cache_path):
        gen_file_list, selected_id_image, selected_id_latent, selected_id = torch.load(cache_path)
    else:
        gen_file_list, selected_id_image, selected_id_latent, selected_id = select_id_latent(files_path,
                                                                                             G,
                                                                                             stage_one_path
                                                                                            )
        torch.save(
                    [
                        gen_file_list,
                        selected_id_image,
                        selected_id_latent,
                        selected_id
                    ],
                     cache_path
                  )
    face_info_from_id = get_face_info(
                                        np.uint8(selected_id_image),
                                        detector
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
                    mask,
                    weights_all,
                    weights,
                    x_pre = None,
                    y_pre = None
                   ):

            l2_loss = self.l2(x, y) * weights_all
            lpips_loss = self.lpips(x,y, is_reduce = False) * weights
            fp_loss = self.fp(x, y, mask)    
            inter_frame_loss = torch.zeros_like(lpips_loss)
            id_loss = self.id_loss(x, y) * 0.0
            ret = {
                     "l2_loss": l2_loss,
                     "lpips_loss": lpips_loss,
                     "fp_loss": fp_loss,
                     "id_loss": id_loss,
                   }
            if x_pre is not None and y_pre is not None:
                inter_frame_loss = self.if_loss(
                                                x,
                                                x_pre.to(x), 
                                                y,
                                                y_pre.to(x),
                                                self.lpips
                                               ) * 2.0 * weights_all
                ret["diff_frame_loss"] = inter_frame_loss

            return ret

    pose_loss_register = PoseLossRegister(config_pose)
    facial_loss_register = FacialLossRegister(config_facial)

    gammas = None
    images_tensor_last = None
    gt_images_tensor_last = None

    style_space_list = []

    optimized_latents = list(filter(lambda x: x.endswith('pt'), os.listdir(stage_three_path)))
    logger.info(f"optimized_latents {optimized_latents}")
    start_idx = len(optimized_latents)
    if start_idx > 0:
        last_tensor_path = os.path.join(stage_three_path, "last_tensor.pt")
        if not os.path.exists(last_tensor_path):
            logger.warn("last tensor not exits, this may harm video stable....")
        else:
            images_tensor_last, gt_images_tensor_last = torch.load(last_tensor_path)

    gen_length = min(len(gen_file_list), 10000) if not DEBUG else 100
    gen_file_list = gen_file_list[:gen_length]
    pbar = tqdm(gen_file_list)

    if VERBOSE:
        t = Timer()
    for ii, _file in enumerate(pbar):
        if ii < start_idx:
            logger.info(f"{ii} processed, pass...")
            continue

        if ii >= len(gen_file_list):
            logger.info("all file optimized, skip to pti.")
            break

        gen_image = cv2.imread(_file)
        assert gen_image is not None, "file not exits, please check."
        gen_image = cv2.cvtColor(gen_image, cv2.COLOR_BGR2RGB)

        # stage 1.5 get face info
        if VERBOSE:
            t.tic("get face info")
        face_info_from_gen = get_face_info(gen_image, detector)
        if VERBOSE:
            t.toc("get face info")
        logger.info("get face info.")

        #stage 2.
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
        torch.save(w_with_pose, os.path.join(stage_two_path, f"{ii+1}.pt"))       
        if DEBUG:
            image_posed = cv2.cvtColor(image_posed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(stage_two_path, f"{ii+1}_pose.jpg"), image_posed * 255.0)
        logger.info("pose optimized..")

        # stage 3.
        style_space_latent, images_tensor_last, gt_images_tensor_last, gammas, image_gen, facial_param = \
                facial_attribute_optimization(w_with_pose, \
                                              gen_image,\
                                              face_info_from_gen,\
                                              facial_loss_register, \
                                              ss_decoder,\
                                              gammas,\
                                              images_tensor_last,\
                                              gt_images_tensor_last \
                                             )
        
        torch.save([x.detach().cpu() for x in style_space_latent], os.path.join(stage_three_path, f"{ii+1}.pt"))       
        torch.save([images_tensor_last, gt_images_tensor_last], os.path.join(stage_three_path, "last_tensor.pt"))

        torch.save([pose_param, facial_param], os.path.join(expressive_param_path, f"attribute_{ii+1}.pt"))
        if DEBUG:
            image_gen = cv2.cvtColor(image_gen, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(stage_three_path, f"{ii+1}_facial.jpg"), image_gen)
        logger.info("facial optimized..")

    # stage 4.
    snapshots = os.path.join(stage_four_path, "snapshots")
    os.makedirs(snapshots, exist_ok = True)
    snapshot_files = os.listdir(snapshots)

    epochs = 100
    pti_or_not = True
    resume_path = None
    if os.path.exists(snapshots) and len(snapshot_files):
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
                                               resume_path = resume_path
                                              )
    logger.info(f"latest model path is {latest_decoder_path}")
    #latest_decoder_path = './results/pivot_001/snapshots/100.pth'
    validate_video_path = os.path.join(save_path, "validate_video.mp4")
    validate_video_gen(
                        validate_video_path,
                        latest_decoder_path,
                        stage_three_path,
                        ss_decoder,
                        len(gen_file_list),
                        face_folder_path
                      )
    logger.info(f"validate video located in {validate_video_path}")

