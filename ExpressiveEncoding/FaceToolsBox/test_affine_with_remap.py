import os
import cv2

import numpy as np

def affine_with_remap(
                        src_image: np.ndarray,
                        matrix: np.ndarray,
                        flow: tuple, 
                        target_size : tuple = (512, 512)
                     ):
    
    #h,w = src_image.shape[:2]
    h, w = 512, 512
    x_linspace = np.linspace(0, w - 1, w)
    y_linspace = np.linspace(0, h - 1, h)
    x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

    def transform(x, y, abc):
        return x * abc[0,0] + y * abc[0,1] + abc[0,2], \
               x * abc[1,0] + y * abc[1,1] + abc[1,2]

    x, y = transform(x_grid, y_grid, matrix)
    
    #x_linspace = np.linspace(0, w - 1, w)
    #y_linspace = np.linspace(0, h - 1, h)
    #x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
    #x, y = x_grid + u, y_grid + v
    """
    loc_0 = np.logical_and(x > 0, y > 0)
    loc_1 = np.logical_and(x < w, y < h)
    loc = np.logical_and(loc_0, loc_1)
    x = x[loc]
    y = y[loc]

    """

    remap_image = cv2.remap(src_image, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
    
    u, v = flow
    x = x + u
    y = y + v
    remap_with_flow_image = cv2.remap(src_image, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR)
    return remap_image, remap_with_flow_image


if __name__ == "__main__":

    import sys

    image = cv2.imread(sys.argv[1])
    matrix = np.load(sys.argv[2])

    flow = np.load(sys.argv[3])

    matrix = np.linalg.inv(matrix)

    remapped_image, remapped_with_flow_image = affine_with_remap(image, matrix, flow)
    cv2.imwrite(sys.argv[4], remapped_image)
    cv2.imwrite(sys.argv[4].replace('.', '_flow.'), remapped_with_flow_image)














