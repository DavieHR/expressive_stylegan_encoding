import shutil
import PIL
import cv2
import imageio
import scipy
import os
import click

import mediapipe as mp
import os.path as osp
import numpy as np

from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from DeepLog import logger

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

th_list = [
    0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87,
    88, 89, 90, 91,
    95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 302, 303, 304, 306,
    307, 308, 310, 311, 312,
    314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 407, 408, 409, 415]

th_list2 = [57,43,106,182,83,18,313,406,335,273,287,410,322,391,393,164,167,165,92,186]
left_eye = [33,7,163,144,145,472,153,154,155,133,173,157,158,470,159,160,161,246,471,468,469]
left_eye2 = [130,25,110,24,23,22,26,112,243,190,56,28,27,29,30,247]
left_eye3 = [226,113,225,224,223,222,221,189,244,233,232,231,230,229,228,31]
right_eye = [382,381,380,477,374,373,390,249,263,466,388,387,386,475,385,384,398,362,473,474,476]
right_eye2 = [463,341,256,252,253,254,339,255,359,467,260,259,257,258,286,414,]
right_eye3 = [464,453,452,451,450,449,448,261,446,342,445,444,443,442,441,413]

def crop_one_image(
                    image: np.ndarray
                  ):
    use_first_eye_to_mouth = True
    e_to_mouth = None
    left_eye_dis = None
    right_eye_dis = None
    h,w = image.shape[:2]
    with mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            raise RuntimeError("no face detected.")
        
        for face_landmarks in results.multi_face_landmarks:
            points_np = np.zeros((len(face_landmarks.landmark), 3), np.float32)
            for idx, point in enumerate(face_landmarks.landmark):
                points_np[idx, 0] = point.x * w
                points_np[idx, 1] = point.y * h
                points_np[idx, 2] = point.z
        lm_eye_left = []
        for key_p_idx in left_eye3:
            lm_eye_left.append(points_np[key_p_idx,:2])
        for key_p_idx in left_eye2:
            lm_eye_left.append(points_np[key_p_idx,:2])
        for key_p_idx in left_eye:
            lm_eye_left.append(points_np[key_p_idx, :2])
        lm_eye_right = []
        for key_p_idx in right_eye3:
            lm_eye_right.append(points_np[key_p_idx,:2])
        for key_p_idx in right_eye2:
            lm_eye_right.append(points_np[key_p_idx,:2])
        for key_p_idx in right_eye:
            lm_eye_right.append(points_np[key_p_idx,:2])
        lm_mouth_outer = []
        for key_p_idx in th_list:
            lm_mouth_outer.append(points_np[key_p_idx,:2])
        for key_p_idx in th_list2:
            lm_mouth_outer.append(points_np[key_p_idx,:2])
        #
        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        if left_eye_dis is None:
            left_eye_dis = points_np[22, 1] + points_np[23, 1] + points_np[24, 1]
            left_eye_dis = left_eye_dis / 3
            left_eye_dis = left_eye_dis - eye_left[1]
        else:
            left_eye_dis_now = points_np[22, 1] + points_np[23, 1] + points_np[24, 1]
            left_eye_dis_now = left_eye_dis_now / 3
            left_eye_dis_now = left_eye_dis_now - eye_left[1]
            if abs(left_eye_dis_now-left_eye_dis) > 1.5:
                eye_left[1] = eye_left[1] - (left_eye_dis -left_eye_dis_now)

        if right_eye_dis is None:
            right_eye_dis = points_np[252, 1] + points_np[253, 1] + points_np[254, 1]
            right_eye_dis = right_eye_dis / 3
            right_eye_dis = right_eye_dis - eye_right[1]
        else:
            right_eye_dis_now = points_np[252, 1] + points_np[253, 1] + points_np[254, 1]
            right_eye_dis_now = right_eye_dis_now / 3
            right_eye_dis_now = right_eye_dis_now - eye_right[1]
            if abs(right_eye_dis_now - right_eye_dis) > 1.5:
                eye_right[1] = eye_right[1] - (right_eye_dis -right_eye_dis_now)

        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        # eye_to_eye = eye_to_eye * 1.1
        mouth_avg = np.mean(lm_mouth_outer, axis=0)
        if use_first_eye_to_mouth:
            if e_to_mouth is None:
                eye_to_mouth = mouth_avg - eye_avg
                e_to_mouth = eye_to_mouth
            else:
                eye_to_mouth = e_to_mouth
        rotate_level = True
        # Choose oriented crop rectangle.
        if rotate_level:
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1
        else:
            x = np.array([1, 0], dtype=np.float64)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
        qsize = np.hypot(*x) * 2

        # # Shrink.
        output_size = 512

        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            quad /= shrink
            qsize /= shrink
            rsize = (int(np.rint(float(image.shape[1]) / shrink)), int(np.rint(float(image.shape[0]) / shrink)))
            image = cv2.resize(image, rsize)
    
        transform_size = 512
        src_points = np.float32(
                [[0, 0], [0, transform_size -  1], [transform_size -  1, transform_size -  1], [transform_size -  1, 0]])
    
        m = cv2.getPerspectiveTransform((quad.astype(np.float32)+0.5), src_points)
        return cv2.warpPerspective(image, m, (512, 512), flags=cv2.INTER_CUBIC)

def crop(
         video_file: str, 
         out_dir: str, 
         use_first_eye_to_mouth: bool = True,
         sigam: int = 2
        ):

    assert os.path.exists(video_file), f"{video_file} not exist."
    os.makedirs(out_dir, exist_ok = True)
    dst_npy = f'{out_dir}/smooth_M.npy'
    out_faces = f'{out_dir}/smooth'
    os.makedirs(out_faces, exist_ok = True)
    lm_list = []
    e_to_mouth = None
    left_eye_dis = None
    right_eye_dis = None
    quads = []

    video_reader = imageio.get_reader(video_file)
    video_meta_info = video_reader.get_meta_data()
    video_length = round(float(video_meta_info['fps']) * float(video_meta_info['duration']))
    pbar_files = tqdm(range(video_length))

    resize_or_not = False
    target_size = None

    for i in pbar_files:
        try:
            image = video_reader.get_next_data()
        except IndexError as s:
            logger.info(f"{video_file} reader EOF.")
            break
        h,w = image.shape[:2]
        with mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(image)
            if not results.multi_face_landmarks:
                raise RuntimeError("no face detected.")
            
            for face_landmarks in results.multi_face_landmarks:
                points_np = np.zeros((len(face_landmarks.landmark), 3), np.float32)
                for idx, point in enumerate(face_landmarks.landmark):
                    points_np[idx, 0] = point.x * w
                    points_np[idx, 1] = point.y * h
                    points_np[idx, 2] = point.z
                lm_list.append(points_np[None, ...])
            lm_eye_left = []
            for key_p_idx in left_eye3:
                lm_eye_left.append(points_np[key_p_idx,:2])
            for key_p_idx in left_eye2:
                lm_eye_left.append(points_np[key_p_idx,:2])
            for key_p_idx in left_eye:
                lm_eye_left.append(points_np[key_p_idx, :2])
            lm_eye_right = []
            for key_p_idx in right_eye3:
                lm_eye_right.append(points_np[key_p_idx,:2])
            for key_p_idx in right_eye2:
                lm_eye_right.append(points_np[key_p_idx,:2])
            for key_p_idx in right_eye:
                lm_eye_right.append(points_np[key_p_idx,:2])
            lm_mouth_outer = []
            for key_p_idx in th_list:
                lm_mouth_outer.append(points_np[key_p_idx,:2])
            for key_p_idx in th_list2:
                lm_mouth_outer.append(points_np[key_p_idx,:2])
            #
            # Calculate auxiliary vectors.
            eye_left = np.mean(lm_eye_left, axis=0)
            eye_right = np.mean(lm_eye_right, axis=0)
            if left_eye_dis is None:
                left_eye_dis = points_np[22, 1] + points_np[23, 1] + points_np[24, 1]
                left_eye_dis = left_eye_dis / 3
                left_eye_dis = left_eye_dis - eye_left[1]
            else:
                left_eye_dis_now = points_np[22, 1] + points_np[23, 1] + points_np[24, 1]
                left_eye_dis_now = left_eye_dis_now / 3
                left_eye_dis_now = left_eye_dis_now - eye_left[1]
                if abs(left_eye_dis_now-left_eye_dis) > 1.5:
                    eye_left[1] = eye_left[1] - (left_eye_dis -left_eye_dis_now)

            if right_eye_dis is None:
                right_eye_dis = points_np[252, 1] + points_np[253, 1] + points_np[254, 1]
                right_eye_dis = right_eye_dis / 3
                right_eye_dis = right_eye_dis - eye_right[1]
            else:
                right_eye_dis_now = points_np[252, 1] + points_np[253, 1] + points_np[254, 1]
                right_eye_dis_now = right_eye_dis_now / 3
                right_eye_dis_now = right_eye_dis_now - eye_right[1]
                if abs(right_eye_dis_now - right_eye_dis) > 1.5:
                    eye_right[1] = eye_right[1] - (right_eye_dis -right_eye_dis_now)

            eye_avg = (eye_left + eye_right) * 0.5
            eye_to_eye = eye_right - eye_left
            # eye_to_eye = eye_to_eye * 1.1
            mouth_avg = np.mean(lm_mouth_outer, axis=0)
            if use_first_eye_to_mouth:
                if e_to_mouth is None:
                    eye_to_mouth = mouth_avg - eye_avg
                    e_to_mouth = eye_to_mouth
                else:
                    eye_to_mouth = e_to_mouth
            rotate_level = True
            # Choose oriented crop rectangle.
            if rotate_level:
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                x /= np.hypot(*x)
                x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
                y = np.flipud(x) * [-1, 1]
                c0 = eye_avg + eye_to_mouth * 0.1
            else:
                x = np.array([1, 0], dtype=np.float64)
                x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
                y = np.flipud(x) * [-1, 1]
                c0 = eye_avg + eye_to_mouth * 0.1


            quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
            qsize = np.hypot(*x) * 2

            # # Shrink.
            output_size = 512

            #shrink = int(np.floor(qsize / output_size * 0.5))
            #if shrink > 1:
            #    resize_or_not = True
            #    target_size = (int(np.rint(float(image.shape[1]) / shrink)), \
            #                   int(np.rint(float(image.shape[0]) / shrink)))
            #    quad /= shrink
            
            quads.append(quad)

    # Transform.
    transform_size = 512
    if len(quads) > 1:
        src_quads = np.array(quads)
        smoothed_src_quads = gaussian_filter1d(src_quads, sigam, axis=0)
    else:
        smoothed_src_quads = quads
    
    src_points = np.float32(
            [[0, 0], [0, transform_size -  1], [transform_size -  1, transform_size -  1], [transform_size -  1, 0]])
    # dst_quad = np.float32(np.array(src_quads) + 0.5)
    logger.info("getting smoothed M and faces ...")
    M_smooth = []
    
    pbar_files = tqdm(range(video_length))
    video_reader.set_image_index(0)
    for i in pbar_files:
        try:
            image = video_reader.get_next_data()
        except IndexError as s:
            break
        out_file = osp.join(out_faces, f'{i}.png')
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if resize_or_not:
            img = cv2.resize(img, target_size, interpolation = cv2.INTER_CUBIC)
        m = cv2.getPerspectiveTransform((smoothed_src_quads[i].astype(np.float32)+0.5), src_points)
        face = cv2.warpPerspective(img, m, (512, 512), flags=cv2.INTER_CUBIC)
        cv2.imwrite(out_file, face)
        M_smooth.append(m.reshape(1, 3, 3))
    M_smooth = np.concatenate(M_smooth, axis=0)
    np.save(dst_npy, M_smooth)


@click.command()
@click.option('--in_dir', help = 'input path')
@click.option('--out_dir', help = 'output path')
@click.option('--use_first_eye_to_mouth', help = 'use_first_eye_to_mouth',default = True)
@click.option('--sigam', help = 'smooth sigam',default = 2)
def main(in_dir: str, 
         out_dir: str, 
         use_first_eye_to_mouth: bool = True,
         sigam: int = 2):
    crop(in_dir, out_dir, use_first_eye_to_mouth, sigam)

if __name__=='__main__':
    main()

