import numpy as np
import cv2 as cv
from copy import deepcopy
from scipy.signal import convolve2d

def harris(img, patch_size, kappa):
    img = deepcopy(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sobel_filter_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])

    sobel_filter_y = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    I_x = convolve2d(img, sobel_filter_x, mode='valid')
    I_y = convolve2d(img, sobel_filter_y, mode='valid')

    I_x2 = np.power(I_x, 2)
    I_y2 = np.power(I_y, 2)
    I_xy = np.multiply(I_x, I_y)

    # grad_mag = np.sqrt(I_x2 + I_y2)
    # cv.imshow("Gradient magnitude", grad_mag.astype(np.uint8))
    # cv.waitKey(0)

    sum_filter = np.ones((patch_size, patch_size))
    sum_I_x2 = convolve2d(I_x2, sum_filter, mode='valid')
    sum_I_y2 = convolve2d(I_y2, sum_filter, mode='valid')
    sum_I_xy = convolve2d(I_xy, sum_filter, mode='valid')

    det_M = (np.multiply(sum_I_x2, sum_I_y2) - np.multiply(sum_I_xy, sum_I_xy))
    trace_M = sum_I_x2 + sum_I_y2
    R = det_M - kappa*np.power(trace_M, 2)
    R = np.pad(R, (patch_size - 1) // 2)
    R[R < 0] = 0

    return R