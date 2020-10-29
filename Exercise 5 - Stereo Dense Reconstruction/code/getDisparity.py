import numpy as np
from scipy.signal import convolve2d
from cython.parallel import prange
from cython import boundscheck, wraparound

import sys
import time

@boundscheck(False)
@wraparound(False)
def getDisparity(left_img, right_img, patch_radius, min_disp, max_disp):
    """% left_img and right_img are both H x W and you should return a H x W
    % matrix containing the disparity d for each pixel of left_img. Set
    % disp_img to 0 for pixels where the SSD and/or d is not defined, and for d
    % estimates rejected in Part 2. patch_radius specifies the SSD patch and
    % each valid d should satisfy min_disp <= d <= max_disp."""

    """ For each pixel in the left image, find the matching pixel in the same
    row in the right image via SSD"""
    H, W = left_img.shape[:2]
    disp_img = np.zeros((H, W))

    start_time = time.monotonic()
    for row in prange(patch_radius, H-patch_radius, schedule='static'):
        for col in range(patch_radius+max_disp, W-patch_radius):

            left_patch = left_img[row-patch_radius:row+patch_radius+1,
                                  col-patch_radius:col+patch_radius+1]

            ssd_arr = []
            for disp in range(min_disp, max_disp+1):
                right_patch = right_img[row-patch_radius:row+patch_radius+1,
                                        col-patch_radius-disp:col+patch_radius+1-disp]
                ssd = np.sum((left_patch-right_patch)**2)
                ssd_arr.append(ssd)

            ssd_arr = np.asarray(ssd_arr)
            min_ssd = np.min(ssd_arr)
            min_idx = np.argmin(ssd_arr)
            best_disp = min_idx + min_disp
            num_similar = np.sum(ssd_arr <= 1.5*min_ssd)

            if (min_disp < best_disp < max_disp) and num_similar <= 3:
                # Sub-pixel refinement
                x = [best_disp-1, best_disp, best_disp+1]
                y = ssd_arr[min_idx-1:min_idx+2]
                p = np.polyfit(x, y, deg=2)
                best_disp = -p[1]/(2*p[0])
                disp_img[row, col] = best_disp
    end_time = time.monotonic()
    print(f"Total Time: {end_time-start_time}")
    return disp_img

