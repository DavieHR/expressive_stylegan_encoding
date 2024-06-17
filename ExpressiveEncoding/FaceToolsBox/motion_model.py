#!/usr/bin/env python
# coding=utf-8
# re-implemention of paper 'Parameterized Models of Image Motion'
#
import os
import sys
import cv2
import math
import pdb
import numpy as np

from DeepLog import logger, Timer
from scipy.ndimage.filters import convolve as filter2

ROOT3 = math.sqrt(3.0)
def spatialAndTimeDerivative(x, y):
    x_kernel = np.array([[-1,1],[-1,1]]) * 0.25
    y_kernel = np.array([[-1,-1],[1,1]]) * 0.25
    t_kernel = np.ones((2,2)) * 0.25

    Ix = filter2(x, x_kernel) + filter2(y, y_kernel)
    Iy = filter2(x, y_kernel) + filter2(y, y_kernel)
    It = filter2(x, -t_kernel) + filter2(y, t_kernel)

    return Ix, Iy, It

def geman_mcclure_ro(x, sigma):
    return x ** 2 / (sigma ** 2 + x ** 2)

def geman_mcclure_phi(x, sigma):
    return x * 2 * (sigma ** 2) / (sigma ** 2 + x ** 2) ** 2

def geman_mcclure_roII(x, sigma):
    return 2 * sigma ** 2 / (sigma ** 2 + x ** 2) ** 2

def X_eight_parameters(x, y, 
                       a0, a1, 
                       a2, a3, 
                       a4, a5, 
                       p0 = 0, p1 = 0):
    """
    """
    u = a0 + a1 * x + a2 * y + p0 * x ** 2 + p1 * x * y
    v = a3 + a4 * x + a5 * y + p0 * x * y + p1 * y ** 2
    return u,v

def get_optical_flow(w, h,
                     parameters
                    ):
    x_linspace = np.linspace(0, w - 1, w) - int(np.floor(w / 2))
    y_linspace = np.linspace(0, h - 1, h) - int(np.floor(h / 2))    
    x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
    
    return X_eight_parameters(x_grid, y_grid, *parameters)

def get_target_xy(u, v):
    h, w = u.shape[:2]
    x_linspace = np.linspace(0, w - 1, w)
    y_linspace = np.linspace(0, h - 1, h)
    x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)
    return x_grid + u, y_grid + v

def warp_image(
               image, 
               a0,a1,a2,
               a3,a4,a5,
               p0=0, p1=0
               ):
    """
    b1 = 1.0 + a1
    b2 = a2
    b4 = a4
    b5 = 1.0 + a5
    det = b1 * b5 - b2 * b4
    b1 /= det
    b2 /= -det
    b4 /= -det
    b5 /= det
    det = b5
    b5 = b1
    b1 = det

    b0 = -b1 * a0 - b2 * a3
    b3 = -b4 * a0 - b5 * a3

    b1 -= 1
    b5 -= 1
    """
    b0,b1,b2,b3,b4,b5 = a0,a1,a2,a3,a4,a5
    h, w = image.shape[:2]
    x_linspace = np.linspace(0, w - 1, w) - w // 2#int(np.floor(w / 2))
    y_linspace = np.linspace(0, h - 1, h) - h // 2#int(np.floor(h / 2))    
    x_grid, y_grid = np.meshgrid(x_linspace, y_linspace)

    u,v = get_optical_flow(w, h, [b0,b1,b2,b3,b4,b5,p0,p1])
    
    x, y = get_target_xy(u, v)
    return cv2.remap(image, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, borderMode = cv2.BORDER_REFLECT101)

def motion_equation(
                    I_x, I_y, I_t,
                    u, v
                    ):
    """
        Returns: 
                equation ans, numpy array, dtype = float64.
    """
    return I_x * u + I_y * v + I_t 

def robust_derivative(
                      I_x, I_y, I_t,
                      u, v, ti,
                      partial_scale, sigma
                     ):
    equation = motion_equation(I_x, I_y, I_t, u, v)
    derivative = geman_mcclure_phi(equation, sigma) * partial_scale
    scale = ti.sum()
    if scale == 0:
        return 0
    return derivative.sum() / scale

def robust_transition(
                      I_x, I_y, I_t,
                      a0, a3,
                      sigma_init, sigma, omega = 1.995, 
                      omask = None, iters = 30, rate = 0.95,
                      eps = 1e-6
                     ):
    """
    """
    u = a0 * np.ones_like(I_x)[omask]
    v = a3 * np.ones_like(I_y)[omask]
    I_x = I_x[omask]
    I_y = I_y[omask]
    I_t = I_t[omask]
    IX2 = (I_x ** 2)
    IY2 = (I_y ** 2)
    s = sigma_init
    for _iter in range(iters):
        Ta0 = ((2 / s ** 2 ) * IX2)
        Ta3 = ((2 / s ** 2 ) * IY2)
        du = robust_derivative(I_x, I_y, I_t, u, v, Ta0, I_x, s)
        a0 -= omega * du
        u = a0 * np.ones_like(I_x)
        dv = robust_derivative(I_x, I_y, I_t, u, v, Ta3, I_y, s)
        a3 -= omega * dv
        v = a3 * np.ones_like(I_x)
        if np.abs(du) < eps and np.abs(dv) < eps:
            logger.info(f"transition regression stop iter is {_iter}")
            break
        s = max(s * rate, sigma)
    return a0, a3


def robust_affine(
                  I_x, I_y, I_t,
                  a0,  a1,  a2, 
                  a3,  a4,  a5,
                  sigma_init, sigma, omega = 1.995, 
                  omask = None,iters = 30, rate = 0.95,
                  eps = 1e-6, p0 = 0, p1 = 0, planar = False
                 ):
    h, w = I_x.shape[:2]
    x_linspace = np.linspace(0, w - 1, w)
    y_linspace = np.linspace(0, h - 1, h)
    x, y = np.meshgrid(x_linspace, y_linspace)
    x = x - w // 2
    y = y - h // 2
    x = x[omask]
    y = y[omask]
    I_x = I_x[omask]
    I_y = I_y[omask]
    I_t = I_t[omask]

    IXx = ((I_x * x) ** 2)
    IXy = ((I_x * y) ** 2)
    IYx = ((I_y * x) ** 2)
    IYy = ((I_y * y) ** 2)

    s = sigma_init
    max_delta = 0
    for _iter in range(iters):
        Ta1 = ((2 / s ** 2 ) * IXx)
        Ta2 = ((2 / s ** 2 ) * IXy)
        Ta4 = ((2 / s ** 2 ) * IYx)
        Ta5 = ((2 / s ** 2 ) * IYy)


        # regression a1
        u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
        u,v = u[omask], v[omask]
        delta = robust_derivative(I_x, I_y, I_t, u, v, Ta1, I_x * x, s)
        a1 -= omega * delta
        max_delta = max(max_delta, np.abs(delta))

        # regression a2
        u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
        u,v = u[omask], v[omask]
        delta = robust_derivative(I_x, I_y, I_t, u, v, Ta2, I_x * y, s)
        a2 -= omega * delta
        max_delta = max(max_delta, np.abs(delta))

        # regression a4
        u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
        u,v = u[omask], v[omask]
        delta = robust_derivative(I_x, I_y, I_t, u, v, Ta4, I_y * x, s)
        a4 -= omega * delta
        max_delta = max(max_delta, np.abs(delta))

        # regression a5
        u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
        u,v = u[omask], v[omask]
        delta = robust_derivative(I_x, I_y, I_t, u, v, Ta5, I_y * y, s)
        a5 -= omega * delta
        max_delta = max(max_delta, np.abs(delta))

        s = max(s * rate, sigma)
        if max_delta < eps:
            logger.info(f"affine regression stop. iter is {_iter}")
            break
    return a1, a2, a4, a5

def robust_planar(
                  I_x, I_y, I_t,
                  a0,  a1,  a2, 
                  a3,  a4,  a5,
                  p0,  p1,
                  sigma_init, sigma, omega = 1.995, 
                  omask = None,iters = 30, rate = 0.95,
                  eps = 1e-6                 
                  ):

    h, w = I_x.shape[:2]
    x_linspace = np.linspace(0, w - 1, w)
    y_linspace = np.linspace(0, h - 1, h)
    x, y = np.meshgrid(x_linspace, y_linspace)
    x = x - w // 2
    y = y - h // 2
    x = x[omask]
    y = y[omask]
    I_x = I_x[omask]
    I_y = I_y[omask]
    I_t = I_t[omask]
    
    I_p0 = I_x * x * x + I_y * x * y
    I_p1 = I_x * x * y + I_y * y * y

    IX2 = (I_x ** 2)
    IY2 = (I_y ** 2)
    IXx = ((I_x * x) ** 2)
    IXy = ((I_x * y) ** 2)
    IYx = ((I_y * x) ** 2)
    IYy = ((I_y * y) ** 2)

    IP02 = (I_p0 ** 2) 
    IP12 = (I_p1 ** 2) 

    s = sigma_init
    max_delta = 0
    for _iter in range(iters):
        Ta0 = ((2 / s ** 2 ) * IX2)
        Ta3 = ((2 / s ** 2 ) * IY2)


        # regression a0
        u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
        u,v = u[omask], v[omask]
        delta = robust_derivative(I_x, I_y, I_t, u, v, Ta0, I_x, s)
        a0 -= omega * delta
        max_delta = max(max_delta, np.abs(delta))

        # regression a0
        u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
        u,v = u[omask], v[omask]
        delta = robust_derivative(I_x, I_y, I_t, u, v, Ta0, I_x, s)
        a0 -= omega * delta
        max_delta = max(max_delta, np.abs(delta))

        if h*w >= 50 * 50:
            Ta1 = ((2 / s ** 2 ) * IXx)
            Ta2 = ((2 / s ** 2 ) * IXy)
            Ta4 = ((2 / s ** 2 ) * IYx)
            Ta5 = ((2 / s ** 2 ) * IYy)
            # regression a1
            u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
            u,v = u[omask], v[omask]
            delta = robust_derivative(I_x, I_y, I_t, u, v, Ta1, I_x * x, s)
            a1 -= omega * delta
            max_delta = max(max_delta, np.abs(delta))

            # regression a2
            u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
            u,v = u[omask], v[omask]
            delta = robust_derivative(I_x, I_y, I_t, u, v, Ta2, I_x * y, s)
            a2 -= omega * delta
            max_delta = max(max_delta, np.abs(delta))

            # regression a4
            u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
            u,v = u[omask], v[omask]
            delta = robust_derivative(I_x, I_y, I_t, u, v, Ta4, I_y * x, s)
            a4 -= omega * delta
            max_delta = max(max_delta, np.abs(delta))

            # regression a5
            u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
            u,v = u[omask], v[omask]
            delta = robust_derivative(I_x, I_y, I_t, u, v, Ta5, I_y * y, s)
            a5 -= omega * delta
            max_delta = max(max_delta, np.abs(delta))

        if h * w >= 100 ** 2:
            Tp0 = ((2 / s ** 2 ) * IP02)
            Tp1 = ((2 / s ** 2 ) * IP12)

            # regression p0
            u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
            u,v = u[omask], v[omask]
            delta = robust_derivative(I_x, I_y, I_t, u, v, Tp0, I_p0, s)
            p0 -= omega * delta
            max_delta = max(max_delta, np.abs(delta))

            # regression p1
            u,v = get_optical_flow(w, h, [a0, a1, a2, a3, a4, a5,p0,p1])
            u,v = u[omask], v[omask]
            delta = robust_derivative(I_x, I_y, I_t, u, v, Tp1, I_p1, s)
            p1 -= omega * delta
            max_delta = max(max_delta, np.abs(delta))


        s = max(s * rate, sigma)
        if max_delta < eps:
            logger.info(f"affine regression stop. iter is {_iter}")
            break
    return a0, a1, a2, a3, a4, a5, p0, p1

    

def robust_region_affine_regression(
                                    I_prevs,
                                    I_nexts,
                                    parameters,
                                    sigma_init,
                                    sigma
                                   ):
    """

    """

    a0,a1,a2,a3,a4,a5,p0,p1 = parameters
    omask = None
    for i, (I_prev, I_next) in enumerate(zip(I_prevs, I_nexts)):
        assert I_prev.ndim == 2 and I_next.ndim == 2, "input must be gray-image."
        h,w = I_prev.shape[:2]
        if i != 0:
            a0 *= 2
            a3 *= 2
            I_prev = warp_image(I_prev, a0, a1, a2, a3, a4, a5)
        I_x, I_y, I_t = spatialAndTimeDerivative(I_prev, I_next)
        if i != 0:
            u, v = get_optical_flow(w,h, [da0,da1,da2,da3,da4,da5])
            omask = np.logical_not(np.abs(motion_equation(I_x, I_y, I_t, u, v)) > (sigma / ROOT3))
        da0,da1,da2,da3,da4,da5 = 0,0,0,0,0,0
        da0, da3 =  robust_transition(I_x, I_y, I_t, 
                                      da0, da3,
                                      sigma_init, sigma,
                                      omask = None#omask
                                     ) 
        if h*w >= 50 * 50:
            da1,da2,da4,da5 = robust_affine(
                                        I_x, I_y, I_t, 
                                        da0,  da1,  da2, 
                                        da3,  da4,  da5,
                                        sigma_init, sigma,
                                        omask = None#omask
                                       )
        a0 += da0
        a1 += da1
        a2 += da2
        a3 += da3
        a4 += da4
        a5 += da5
    return [a0,a1,a2,a3,a4,a5], omask

def robust_planar_regression(
                            I_prevs,
                            I_nexts,
                            parameters,
                            sigma_init,
                            sigma
                           ):
    """

    """

    a0,a1,a2,a3,a4,a5,p0,p1 = parameters
    omask = None
    for i, (I_prev, I_next) in enumerate(zip(I_prevs, I_nexts)):
        assert I_prev.ndim == 2 and I_next.ndim == 2, "input must be gray-image."
        h,w = I_prev.shape[:2]
        if i != 0:
            a0 *= 2
            a3 *= 2
            I_prev = warp_image(I_prev, a0, a1, a2, a3, a4, a5, p0, p1)
        I_x, I_y, I_t = spatialAndTimeDerivative(I_prev, I_next)
        if i != 0:
            u, v = get_optical_flow(w,h, [da0,da1,da2,da3,da4,da5,dp0,dp1])
            omask = np.logical_not(np.abs(motion_equation(I_x, I_y, I_t, u, v)) > (sigma / ROOT3))
        da0,da1,da2,da3,da4,da5,dp0,dp1 = 0,0,0,0,0,0,0,0
        da0,da1,da2,da3,da4,da5,dp0,dp1 = robust_planar(
                                    I_x, I_y, I_t, 
                                    da0,  da1,  da2, 
                                    da3,  da4,  da5,
                                    dp0, dp1,
                                    sigma_init, sigma,
                                    omask = None#omask
                                   )
        a0 += da0
        a1 += da1
        a2 += da2
        a3 += da3
        a4 += da4
        a5 += da5
        p0 += dp0
        p1 += dp1

    return [a0,a1,a2,a3,a4,a5,p0,p1], omask

def solve( 
            I_prev,
            I_next,
            planar = False
         ):
    n_iters = 30
    theta = 1e-6
    omega = 1.995
    sigma = 15 * ROOT3
    sigma_init = 20 * ROOT3
    parameters = [
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0
                 ]
    _long_length = max(I_next.shape[:2])

    log2 = lambda x: math.log(x) / math.log(2)

    # paramid.
    n_level = int(log2(_long_length)) - int(log2(16))

    images_next_in_paramid = [I_next]

    image = I_next
    for _ in range(n_level - 1):
        image = cv2.pyrDown(image)
        images_next_in_paramid.append(image)

    images_prev_in_paramid = [I_prev]
    image = I_prev
    for _ in range(n_level - 1):
        image = cv2.pyrDown(image)
        images_prev_in_paramid.append(image)

    images_prev_in_paramid = images_prev_in_paramid[::-1]
    images_next_in_paramid = images_next_in_paramid[::-1]
    h, w = I_prev.shape[:2]
    # backward
    #u,v = get_optical_flow(w,h,[-0.758066,-0.010512,-0.001354,2.140018,-0.001423,-0.003781])
    #return [-0.758066,-0.010512,-0.001354,2.140018,-0.001423,-0.003781], np.zeros((h,w), dtype = np.uint8)
    #u,v = get_optical_flow(w, h, [0.763180, 0.010625, 0.001374, -2.147050, 0.001443, 0.003797])
    #return get_optical_flow(w, h, [-3.59432, 0.003995, 0.002025, -3.527813, -0.000553, 0.003694])
    func = robust_planar_regression if planar else robust_region_affine_regression
    parameters, omask = func(
                             images_prev_in_paramid,
                             images_next_in_paramid,
                             parameters,
                             sigma_init,
                             sigma
                            )
    return parameters, omask

def ba_optical_flow(
                    refImage,
                    keyImage,
                    planar = False
                   ):
    return solve(refImage, keyImage, planar)

def get_masked_image(x):
    image = x.copy()
    image[165:278, 129:242] = 0
    image[165:279, 268:382] = 0
    image[310:454, 184:328] = 0
    return image

def affineWarpWithKey(
                      refImage,
                      keyImage,
                      planar = False,
                      threshold = 30.0,
                      is_fast = False
                     ):

    if is_fast:
        target_size = 256
        scale = max(refImage.shape[0] / target_size, refImage.shape[1] / target_size)
        refImage_resize = cv2.resize(refImage, (0,0), fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_CUBIC)
        keyImage_resize = cv2.resize(keyImage, (0,0), fx = 1 / scale, fy = 1 / scale, interpolation = cv2.INTER_CUBIC)
        keyImage_gray = cv2.cvtColor(keyImage_resize, cv2.COLOR_BGR2GRAY).astype(np.float64)
        refImage_gray = cv2.cvtColor(refImage_resize, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        keyImage_gray = cv2.cvtColor(keyImage, cv2.COLOR_BGR2GRAY).astype(np.float64)
        refImage_gray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY).astype(np.float64)

    keyImage_gray = get_masked_image(keyImage_gray)
    refImage_gray = get_masked_image(refImage_gray)
    pre_process = lambda x: x #lambda x: (x - x.min()) / (x.max() - x.min()) * 255.0
    t = Timer()
    t.tic("affine_flow")
    parameters, _ = ba_optical_flow(pre_process(refImage_gray), pre_process(keyImage_gray), planar)

    if is_fast:
        parameters[0] *= scale
        parameters[3] *= scale

    warpedImage = warp_image(refImage, *parameters)
    
    t.toc("affine_flow")
    error = np.sqrt(((get_masked_image(warpedImage) - get_masked_image(keyImage)) ** 2).mean())
    error_bench = np.sqrt(((get_masked_image(refImage) - get_masked_image(keyImage)) ** 2).mean())
    logger.info(f"mse is {error} benchmark {error_bench}")
    return warpedImage, error <= np.sqrt(threshold), error - error_bench

if __name__ == "__main__":
    keyFramePath = sys.argv[1]
    refFramePath = sys.argv[2]
    is_planar = int(sys.argv[3])
    is_fast = int(sys.argv[4])

    keyImage = cv2.imread(keyFramePath)
    refImage = cv2.imread(refFramePath)

    warpedImage, flag, _ = affineWarpWithKey(refImage, keyImage, planar = is_planar, is_fast = is_fast)

    postfix = ".png"
    if is_planar:
        postfix = "_planar" + postfix
    if is_fast:
        postfix = "_fast" + postfix

    cv2.imwrite(f"warpedImage{postfix}", warpedImage)
