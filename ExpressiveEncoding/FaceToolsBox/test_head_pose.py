import os
import sys
sys.path.insert(0, os.getcwd())

import cv2

from DeepLog import logger, Timer
from test_ldm import get_detector, infer, imageio

if __name__ == "__main__":
    detector = get_detector()
    debug = False
    FRAMES = 100
    path = "/data1/wanghaoran/Amemori/third_party/MotionModel/backup/Regression/video_dataset/face.mp4" 
    save_path = "pose.mp4"
    reader_init = imageio.get_reader(path)
    meta_info = reader_init.get_meta_data()
    fps = meta_info["fps"]
    writer = imageio.get_writer(save_path, fps = fps)
    i = 0
    while(1):
        if debug:
            if i> FRAMES:
                break
        try:
            image = reader_init.get_next_data()
        except Exception as e:
            break
        res = infer(detector, image)
        rotation_transform = res.facial_transformation_matrixes[0][:3,:3]
        #get angle
        angle, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_transform)
        x_angle, y_angle, z_angle = angle
        image = cv2.putText(image, f"x angle is {x_angle}", (51,51), cv2.FONT_HERSHEY_SIMPLEX, 0.2,(255,0,0))
        image = cv2.putText(image, f"y angle is {y_angle}", (51,61), cv2.FONT_HERSHEY_SIMPLEX, 0.2,(255,0,0))
        image = cv2.putText(image, f"z angle is {z_angle}", (51,71), cv2.FONT_HERSHEY_SIMPLEX, 0.2,(255,0,0))
        i += 1
        writer.append_data(image)
    
