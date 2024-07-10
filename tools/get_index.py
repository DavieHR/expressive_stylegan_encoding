import os
import sys
sys.path.insert(0, os.getcwd())
import cv2
import torch
import click
import pickle
import numpy as np

from typing import List
from ExpressiveEncoding.train import gen_masks, get_face_info, get_detector,\
                                      stylegan_path, StyleSpaceDecoder, alphas as ALPHAS, \
                                      get_mask_by_region

from ExpressiveEncoding.loss import LPIPS
from DeepLog import logger

from tqdm import tqdm

def debug_tensor2image( 
                        tensor: torch.Tensor,
                        save_path: str
                      ):

    image = tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    image = (image + 1) * 0.5 * 255
    ret = cv2.imwrite(save_path, image)
    
def update_alpha(
                    latents: list,
                    alphas: list
                ):

     dlatents_tmp = [dlatent.clone() for dlatent in latents]
     count = 0
     # first 5 elements.
     for latent, alpha in zip(dlatents_tmp, alphas):
         latent += alpha
     return dlatents_tmp

def get_alpha_deltas(
                      random_batchsize: int,
                      ss_decoder: object
                    ):
    
    device = "cuda:0"

    z_latent = torch.randn((random_batchsize,512)).to(device)

    w_latent = ss_decoder.mapping(z_latent, 0)
    w_space_latents = ss_decoder.get_style_space(w_latent)

    deltas = []

    for latent in w_space_latents:
        deltas.append(latent.std(dim = 0, keepdim = True))
    return deltas


def get_scores(
                id_ss_latent: List[torch.Tensor],
                gt_image: torch.Tensor,
                mask: torch.Tensor,
                ss_decoder: object,
                alpha_deltas: list,
                list_path: list = None
              ):
    lpips = LPIPS()
    alphas = [torch.zeros_like(latent) for latent in id_ss_latent]
    perbulents = [-24, -16, -8, 8, 16, 24]
    scores = [torch.zeros_like(latent) for latent in id_ss_latent]

    if list_path is not None:
        if isinstance(list_path, str):
            with open(list_path, "rb") as f:
                _list = pickle.load(f)
                if len(_list[0]) > 2:
                    _list = [(x[0], x[1]) for x in _list]
        else:
            _list = list_path
    else:
        _list = []
        for l, alpha in enumerate(ALPHAS[5:8]):
            (k, cs) = alpha
            for c in cs:
                _list.append((k, c))

    #part = ALPHAS[5:8]
    #for l, alpha in enumerate(alphas):
    """
    if l not in [x[0] for x in part]:
        continue
    """
        #if alpha.ndim > 1:
        #    alpha = alpha[0]
        #for c, unit in enumerate(alpha):
    #for i, (l, channels) in enumerate(part):
    #    for j, c in enumerate(channels):
    pbar = tqdm(_list)

    for i, (l, c) in enumerate(pbar):
            pre_value = 0.0
            for perbulent in perbulents:
                diff = perbulent - pre_value
                alphas[l][:, c] = alpha_deltas[l][:, c] * perbulent
                updated_latent = update_alpha(id_ss_latent, alphas)
                with torch.no_grad():
                    gen_image = ss_decoder(updated_latent)
                    scores[l][:, c] += lpips(gen_image * mask, gt_image * mask) / (diff)
                #if l ==0 and c == 0:
                #logger.info("save...")
                #debug_tensor2image(gen_image * mask, f"./image_{perbulent}_{l}_{c}_gen.jpg")
                #debug_tensor2image(gt_image * mask, f"./image_{perbulent}_{l}_{c}_gt.jpg")

                pre_value = perbulent
            scores[l][:, c] /= len(perbulents)
            alphas[l][:,c] = 0.0
    return scores

def get_index(
                id_latent: torch.Tensor,
                id_landmarks: np.ndarray,
                ss_decoder: object,
                save_path: str,
                list_path: str = None
             ):
    device = "cuda:0"
    id_latent = id_latent.to(device)
    ss_decoder.to(device)

    id_ss_latent = ss_decoder.get_style_space(id_latent)
    with torch.no_grad():
        gt_image = ss_decoder(id_ss_latent)
    deltas = get_alpha_deltas(100_000, ss_decoder)
    
    mask = gen_masks(id_landmarks, gt_image.detach().squeeze().cpu().permute(1,2,0).numpy())["chin"]
    #mask = (mask + mask_chin)
    #mask[mask > 0] = 1


    scores_region = get_scores(
                         id_ss_latent,
                         gt_image,
                         mask,
                         ss_decoder,
                         deltas,
                         list_path
                       )

    scores = [[(l, c, y) for c, y in enumerate(x.detach().cpu().numpy().tolist()[0])] for l, x in enumerate(scores_region)]
    scores_new = []
    for score in scores:
        scores_new += score
    #logger.info(sorted(scores_new, key = lambda x: x[2]))
    sorted_scores = [x for x in scores_new if x[2] > 0.9e-3]
    logger.info(sorted(sorted_scores, key = lambda x: x[2]))
    sorted_scores = [(x[0], x[1]) for x in sorted(sorted_scores, key = lambda x: x[2])]
    
    _ , mask = get_mask_by_region()
    mask = cv2.resize(mask ,(1024, 1024))
    mask = torch.tensor(mask).to(device).unsqueeze(0)
    mask = mask.permute(0,3,1,2)

    scores_fixed = get_scores(
                         id_ss_latent,
                         gt_image,
                         mask,
                         ss_decoder,
                         deltas,
                         sorted_scores
                       )
    
    scores = [[(l, c, y) for c, y in enumerate(x.detach().cpu().numpy().tolist()[0])] for l, x in enumerate(scores_fixed)]
    scores_new = []
    for score in scores:
        scores_new += score
    #logger.info(sorted(scores_new, key = lambda x: x[2]))
    sorted_scores = [x for x in scores_new if x[2] != 0] #if x[2] < 1e-4 and x[2] != 0]
    logger.info(sorted(sorted_scores, key = lambda x: x[2] ,reverse = True))

    """
    torch.save(scores, save_path)
    """

@click.command()
@click.option("--cache_path", default = None)
@click.option("--save_path", default = None)
@click.option("--list_path", default = None)
@click.option("--decoder_path", default = None)
def invoker(
            cache_path,
            save_path,
            list_path,
            decoder_path
            ):

    gen_file_list, selected_id_image, selected_id_latent, selected_id = torch.load(cache_path)

    detector = get_detector()

    face_info_from_id = get_face_info(
                                        np.uint8(selected_id_image),
                                        detector
                                     )
    ss_decoder = StyleSpaceDecoder(stylegan_path)

    if decoder_path is not None:
        logger.info(f"load state dict from {decoder_path}.")
        state_dict = torch.load(decoder_path)
        ss_decoder.load_state_dict(state_dict, False)

    get_index(
              selected_id_latent,
              face_info_from_id.landmarks,
              ss_decoder,
              save_path,
              list_path
             )

if __name__ == "__main__":
    invoker()





