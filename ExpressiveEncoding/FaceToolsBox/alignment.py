import os
import scipy
from skimage import transform
from easydict import EasyDict as edict
from scipy.ndimage import gaussian_filter1d
from .test_ldm import (get_detector, imageio, cv2, 
                    infer, logger, np, 
                    affineWarpWithKey, get_landmarks_from_mediapipe_results, 
                    need_to_warp, mp)


from .motion_model import get_masked_image

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

eyes_ldm = left_eye + left_eye2 + left_eye3 + \
           right_eye + right_eye2 + right_eye3

def convert_poins(ldms, matrix):
    return np.matmul(ldms, matrix[:2,:2]) + matrix[2,:2]

def stylegan_alignment(image, res, res_reference = None):
    h, w = image.shape[:2]

    if isinstance(res, mp.tasks.vision.FaceLandmarkerResult):
        res = get_landmarks_from_mediapipe_results(res)
        if res_reference is not None:
            res_reference = get_landmarks_from_mediapipe_results(res_reference)
            res_reference[:,0] *= w
            res_reference[:,1] *= h
        res[:,0] *= w
        res[:,1] *= h
        res_copy = res.copy()

    if res_reference is not None:
        eyes_ldm = left_eye + left_eye2 + left_eye3 + \
                   right_eye + right_eye2 + right_eye3
        # res alignment with res_reference.
        remaining_index = list(set(list(range(len(res)))) - set(eyes_ldm + th_list + th_list2)) # exclude eye and mouth points.
        transform_instance = transform.SimilarityTransform()
        residual = transform_instance.residuals(res_reference[remaining_index,:], res[remaining_index,:])


        transform_instance.estimate(res_reference[remaining_index,:], res[remaining_index,:])
        residual_after = transform_instance.residuals(res_reference[remaining_index,:], res[remaining_index,:])
        logger.info(f"{residual.mean()} {residual_after.mean()}")
        m_to_aligned = transform_instance.params.T # 3x3 
        #res[eyes_ldm, :] = np.matmul(res_reference[eyes_ldm, :] - m_to_aligned[2,:2], iM) 
        res[eyes_ldm, :] = np.matmul(res_reference[eyes_ldm, :], m_to_aligned[:2,:2]) + m_to_aligned[2,:2] 
    ldm = res

    left_eye_points = np.concatenate((ldm[left_eye],ldm[left_eye2],ldm[left_eye3]), axis = 0)
    right_eye_points = np.concatenate((ldm[right_eye],ldm[right_eye2],ldm[right_eye3]), axis = 0)
    mouth_points = np.concatenate((ldm[th_list],ldm[th_list2]), axis = 0)

    eye_left = np.mean(left_eye_points, axis=0)
    eye_right = np.mean(right_eye_points, axis=0)

    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = (eye_right - eye_left) * 1.1
    mouth_avg = np.mean(mouth_points, axis=0)
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c0 = eye_avg + eye_to_mouth * 0.1

    quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
    qsize = np.hypot(*x) * 2

    # # Shrink.
    output_size = 512
    transform_size = 512
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    src_points = np.float32(
            [[0, 0], [0, transform_size -  1], [transform_size -  1, transform_size -  1], [transform_size -  1, 0]])
    m = cv2.getPerspectiveTransform((quad.astype(np.float32)+0.5), src_points)
    #face = cv2.warpPerspective(image, m, (512, 512), flags=cv2.INTER_CUBIC)
    return m

def get_euler_angle(res):
    rotation_transform = res.facial_transformation_matrixes[0][:3,:3]
    #get angle
    angle, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_transform)
    return angle

if __name__ == "__main__":
    detector = get_detector()
    debug = True
    is_planar = False
    is_fast = False
    is_compensate = True
    is_debug_mask = False
    is_debug_stylegan_alignment = True
    is_affine_first = False
    #path = "/data1/wanghaoran/Amemori/third_party/MotionModel/backup/Regression/hard_female.mp4" 
    #path = "/data1/chenlong/0517/dingzhi/lehehe/data/20230805_115249921/20230805_115249921.mp4" 
    #path = "/data1/chenlong/0517/dingzhi/lehehe/data/20230805_115249921/20230805_115249921.mp4" 
    #path = "face1.mp4" 
    path = "/data1/chenlong/online_model_set/speed/exp_speed_v3/release_v3/QjXSCiOY/templates/video.mp4"

    postfix = ".mp4"

    if is_debug_mask:
        postfix = "_masked" + postfix
    if is_planar:
        postfix = "_planar" + postfix
    if is_fast:
        postfix = "_fast" + postfix
    if is_debug_stylegan_alignment:
        postfix = "_stylegan_alignment" + postfix
    if is_compensate:
        postfix = "_compensate" + postfix
    if is_affine_first:
        postfix = "_affine_first" + postfix

    save_path = "output" + postfix
    START_FRAMES = 585 #3 * 25 if is_debug_stylegan_alignment else 0
    FRAMES =  START_FRAMES + 10 #10 if not debug else 10
    reader_init = imageio.get_reader(path) 
    meta_info = reader_init.get_meta_data()
    fps = meta_info["fps"]
    key_frames = []

    i, j = START_FRAMES, START_FRAMES
    need_compensate_frames = dict(zip([i for i in range(1)], [0 for _ in range(1)]))
    face_infos = []

    with imageio.get_reader(path) as reader, \
         imageio.get_writer(save_path, fps = fps) as writer:

        reader.set_image_index(START_FRAMES)

        # blink check.
        length_video = 0
        matrixs = []
        pad = 3
        need_compensate_frames = dict()
        need_compensate_frames[0] = dict(
                                         compensate = False,
                                         next_idx = 0,
                                         ldm = None
                                        )
        while(1):
            if debug:
                if i> FRAMES:
                    logger.info(f"Finished.")
                    break
            try:
                image = reader.get_next_data()
                logger.info(f"{i}th processing.")
            except Exception as e:
                logger.info(f"Finished.")
                break
            res = infer(detector, image)
            face_infos.append(res)
            if len(res.face_blendshapes) <= 0:
                m = None
                matrixs.append(m)
                i += 1
                continue
                
            if is_debug_stylegan_alignment:
                if (i not in need_compensate_frames) or (i in need_compensate_frames and not need_compensate_frames[i]["compensate"]):
                    need_compensate_frames[i - START_FRAMES] = dict(
                                                      compensate = False,
                                                      prev_idx = i - START_FRAMES,
                                                      next_idx = i - START_FRAMES
                                                    )
                if need_to_warp(res):
                    if (i - START_FRAMES not in need_compensate_frames) or (i - START_FRAMES in need_compensate_frames and not need_compensate_frames[i - START_FRAMES]["compensate"]):
                        start = max(i - START_FRAMES - pad, 0)
                        end = i - START_FRAMES + pad
                        next_idx = max(start - 1, 0)
                        for _j in range(start, end + 1):
                            logger.info(f"blink compensate frames is {j} with {next_idx}")
                            need_compensate_frames[_j] = dict(compensate = True, prev_idx = start - 1, next_idx = end + 1)

                m = stylegan_alignment(image, res)

                matrixs.append(m)
            i += 1
            length_video += 1
        logger.info(f"video length is {length_video}")
        matrixs = np.array(matrixs)
        matrixs = gaussian_filter1d(matrixs, 2, axis=0)

        i = START_FRAMES
        reader.set_image_index(START_FRAMES)
        while(1):
            if debug:
                if i> FRAMES:
                    logger.info(f"Finished.")
                    break
            try:
                image = reader.get_next_data()
                image_origin = image.copy()
                logger.info(f"{i}th processing.")
            except Exception as e:
                logger.info(f"Finished.")
                break

            if is_debug_stylegan_alignment:
                matrix = matrixs[i - START_FRAMES]
                if matrix is None:
                    i += 1
                    continue
                if is_compensate and need_compensate_frames[i - START_FRAMES]["compensate"]:
                    next_j = need_compensate_frames[i - START_FRAMES]["next_idx"]
                    prev_j = need_compensate_frames[i - START_FRAMES]["prev_idx"]
                    logger.info(f"{i}th compensate with {prev_j}th and {next_j}th frames")
                    matrix = stylegan_alignment(image, face_infos[i - START_FRAMES], face_infos[prev_j])
                image = cv2.warpAffine(image, matrix[:2,:], (512, 512), flags=cv2.INTER_CUBIC)
                #writer.append_data(aligned_by_stylegan)
                #i += 1
                #continue
            
            image_copy = image.copy()
            res = face_infos[i - START_FRAMES]
            key_frames.append(dict(frame = image_copy, angle = get_euler_angle(res)))
            error = 0.0
            while(i > j):
                keyFrame = key_frames[j - START_FRAMES]["frame"]
                angle = key_frames[j - START_FRAMES]["angle"]
                angle_current = get_euler_angle(res)
                change_yaw = np.abs(angle[1] - angle_current[1])
                change_pitch = np.abs(angle[2] - angle_current[2])

                threshold = 45.0
                if change_pitch > 2 or change_yaw > 2:
                    threshold = 30.0
                logger.info(f"current threshold {threshold}")
                image_warped, flag, error, flow = affineWarpWithKey(image, keyFrame, is_planar, threshold, is_fast = is_fast, matrix = matrixs[i] if is_affine_first else None, image_origin = image_origin)
                if flag:
                    logger.info(f"key {j}. affined")
                    image = image_warped
                    break
                logger.info(f"find a new key.")
                j += 1
            #cv2.imwrite(f"warpedFrame_{i}_0.jpg",  image_copy)
            #cv2.imwrite(f"warpedFrame_{i}_1.jpg",  key_frames[j]["frame"])
            cv2.imwrite(f"warpedFrame_{i}_2_{postfix}.jpg",  image[...,::-1])
            #logger.info(f"j is {j}")
            #image = cv2.putText(image, f"affine_with_key_{j}th_{error:.5}", (51,51), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            #image_copy = cv2.putText(image_copy, "origin", (51,51), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            #image_key = cv2.putText(key_frames[j]["frame"], f"key_frame_{j}", (51,51), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            #writer_list = [image_copy, image, image_key]
            #if is_debug_mask:
            #    writer_list.append(get_masked_image(image))
            writer_list = [image]
            writer.append_data(np.concatenate(tuple(writer_list), axis = 1))
            i += 1
