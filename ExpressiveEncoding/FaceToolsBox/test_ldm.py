import os
import sys
import cv2
import imageio
import pdb
import numpy as np
import mediapipe as mp

from  mediapipe.tasks import python
from  mediapipe.tasks.python import vision
from DeepLog import logger
from .motion_model import affineWarpWithKey, get_masked_image

th_list = [
            0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87,
            88, 89, 90, 91,
            95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 302, 303, 304, 306,
            307, 308, 310, 311, 312,
            314, 315, 316, 317, 318, 319, 320, 321, 324, 325, 375, 402, 403, 404, 405, 407, 408, 409, 415
          ]

# 13 14 point.

def get_detector():
    absolute_path = os.path.dirname(os.path.realpath(__file__))
    base_options = python.BaseOptions(model_asset_path=os.path.join(absolute_path, 'face_landmarker_v2_with_blendshapes.task'))
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

def infer(detector, image):
    image_mp = mp.Image(image_format = mp.ImageFormat.SRGB, data = image)
    return detector.detect(image_mp)

def get_landmarks_from_mediapipe_results(res):
    landmarks_not_parsed = res.face_landmarks[0]
    ldms = np.zeros((len(landmarks_not_parsed), 2), dtype = np.float32)
    for i,item in enumerate(landmarks_not_parsed):
        ldms[i,:] = np.array([item.x,item.y], dtype = np.float32).reshape(1,2)

    return ldms
### image color is RGB

def draw_landmarks(
                   image,
                   landmark,
                   select_ids = None
                  ):
    image_resize = cv2.resize(image, (2048,2048))
    for idx, point in enumerate(landmark[select_ids]):
        if select_ids is None:
            id_to_show = idx
        else:
            id_to_show = select_ids[idx]
        image_resize = cv2.circle(image_resize, np.int32(point * 4).tolist(), 1, (0, 0, 255))
        image_resize = cv2.putText(image_resize, f"{id_to_show}", np.int32(point * 4).tolist(),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,0,0))
    cv2.imwrite("landmark_image.png", image_resize[...,::-1])

def need_to_warp(res):
    return (res.face_blendshapes[0][9].score > 0.5 or res.face_blendshapes[0][10].score > 0.5)

def get_nearest_key_index(i,
                          keys
                         ):
    key = keys.pop()
    while(key > i):
        key = keys.pop()
    return key
    

if __name__ == "__main__":
    detector = get_detector()
    debug = False
    is_planar = False
    path = "/data1/wanghaoran/Amemori/third_party/MotionModel/backup/Regression/video_dataset/face.mp4" 
    save_path = "output_every_blink_planar.mp4" if is_planar else "output_every_blink.mp4" 
    FRAMES = 100
    reader_init = imageio.get_reader(path) 
    meta_info = reader_init.get_meta_data()
    fps = meta_info["fps"]
    compensate_mid_points = []
    key_frames_id = []
    key_frames = []
    i = 0
    need_compensate_frames = dict(zip([i for i in range(1)], [0 for _ in range(1)]))
    pad = 5
    while(1):
        if debug:
            if i> FRAMES:
                break
        try:
            image = reader_init.get_next_data()
        except Exception as e:
            break
        print(type(image), image)
        res = infer(detector, image)
        if need_to_warp(res):
            if (i not in need_compensate_frames) or (i in need_compensate_frames and need_compensate_frames[i] == 0):
                start = max(i - pad, 0)
                end = i + pad
                for j in range(start, end + 1):
                    need_compensate_frames[j] = 1

        elif i not in need_compensate_frames:
            need_compensate_frames[i] = 0
        i += 1
    logger.info(need_compensate_frames)
    reader_init.set_image_index(0) 

    i = 0
    # find key points.
    while(1):
        if debug:
            if i > FRAMES:
                break
        try:
            image = reader_init.get_next_data()
        except Exception as e:
            break

        if need_compensate_frames[i] == 0:
            key_frames.append(image)
        i += 1
        
    key_frames = key_frames[::-1]

    total_key_frames = len(key_frames)

    error_cal = lambda x, y: ((x - y) ** 2).mean()

    with imageio.get_reader(path) as reader, \
         imageio.get_writer(save_path, fps = fps) as writer:
        keyFrame = None
        i = 0
        while(1):
            if debug:
                if i > FRAMES:
                   break
            try:
                image = reader.get_next_data()
            except Exception as e:
                break
            image_copy = image.copy()
            h,w = image.shape[:2]
            if need_compensate_frames[i]:
                flag = False
                while(not flag):
                    if keyFrame is None:
                        keyFrame = key_frames.pop()
                    image, flag = affineWarpWithKey(image, keyFrame, is_planar)
                    if not flag:
                        keyFrame = key_frames.pop()
                    logger.info(f"{i}th frame warped select key {total_key_frames - len(key_frames)}th frame.")

            image = cv2.putText(image, "affine_estimate", (51,51), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
            image_copy = cv2.putText(image_copy, "origin", (51,51), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
            writer.append_data(np.concatenate((image_copy, image), axis = 1))
            #writer.append_data(image)
            i += 1
        
