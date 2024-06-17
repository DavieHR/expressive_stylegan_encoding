"""get alignmented face.
"""
from typing import List

import os
import sys
sys.path.insert(0, os.getcwd())
import tqdm
import click
import torch
import re

import numpy as np

from ExpressiveEncoding.train import validate_video_gen, \
                                     StyleSpaceDecoder, stylegan_path, PoseEdit, \
                                     get_detector, get_face_info


from scipy.ndimage import gaussian_filter1d

def stable_pose(
                selected_id_image: np.ndarray,
                selected_id_latent: torch.tensor,
                pose_path:str
               ) -> List[torch.tensor]:
    
    # init pose edit network.
    pose_edit = PoseEdit()

    # init detector for get_face_info.
    detector = get_detector()

    # read all pose to memory.
    pose_list = [torch.load(os.path.join(pose_path, x)).detach().cpu().numpy() for x in sorted(os.listdir(pose_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))]

    # stable pose.
    pose_stabled = gaussian_filter1d(np.array(pose_list), 1, axis = 0)
    p_bar = tqdm.tqdm(range(len(pose_list)))
    # get stabled latent.
       
    face_info_from_id = get_face_info(
                                        np.uint8(selected_id_image),
                                        detector
                                     )

    yaw, pitch = face_info_from_id.yaw, face_info_from_id.pitch
    with torch.no_grad():
        id_zflow = pose_edit(selected_id_latent, yaw, pitch)

    latents = []
    
    for i in p_bar:
        _pose = pose_stabled[i]

        yaw, pitch = _pose[0], _pose[1]
        yaw = torch.tensor([yaw]).type(torch.FloatTensor)
        pitch = torch.tensor([pitch]).type(torch.FloatTensor)
        with torch.no_grad():
            w = pose_edit(id_zflow, 
                          yaw,
                          pitch,
                          True)
        latents.append(w)

    return latents

def get_pose_video(
                    from_path: str,
                    to_path: str,
                    pose_path: str = None
                  ):
    ss_decoder = StyleSpaceDecoder(stylegan_path)
    face_folder_path = os.path.join(from_path, "data", "smooth")
    e4e_path = os.path.join(from_path, "cache.pt")
    gen_file_list, selected_id_image, selected_id_latent, selected_id = torch.load(e4e_path)
    if pose_path is None:
        pose_path = os.path.join(from_path, "pose")
    stabled_pose_latents = stable_pose(selected_id_image, selected_id_latent, pose_path)
    stabled_pose_latent_path = os.path.join(to_path, "pose_stabled")
    os.makedirs(stabled_pose_latent_path, exist_ok = True)
    for i in range(len(gen_file_list)):
        torch.save(stabled_pose_latents[i], os.path.join(stabled_pose_latent_path, f"{i + 1}.pt"))

    """
    snapshots = os.path.join(from_path, "pti", "snapshots")
    snapshot_files = os.listdir(snapshots)
    snapshot_paths = sorted(snapshot_files, key = lambda x: int(x.split('.')[0]))
    latest_decoder_path = os.path.join(snapshots, snapshot_paths[-1])
    to_path_video = os.path.join(to_path, "video.mp4")

    validate_video_gen(
                        to_path_video,
                        latest_decoder_path,
                        stabled_pose_latents,
                        ss_decoder,
                        len(gen_file_list),
                        face_folder_path
                      )
    """

@click.command()
@click.option('--from_path')
@click.option('--to_path')
@click.option('--pose_path', default = None)
def invoker(
             from_path,
             to_path,
             pose_path
           ):
    return get_pose_video(from_path, to_path, pose_path)

if __name__ == '__main__':
    invoker()
