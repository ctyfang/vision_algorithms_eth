import numpy as np
from scipy.signal import convolve2d
from copy import deepcopy

def selectKeypoints(scores, num, r):
    """% Selects the num best scores as keypoints and performs non-maximum
        % supression of a (2r + 1)*(2r + 1) box around the current maximum.

        Return keypoints in (row, col) format """

    scores = deepcopy(scores)

    keypoints = []
    for i in range(num):
        max_idx = np.argpartition(scores, -2, axis=None)[-1]
        row, col = np.unravel_index(max_idx, scores.shape)
        check_val = scores[row, col]
        keypoints.append([row, col])

        # NMS
        patch = scores[max(row-r,0):min(row+r+1,scores.shape[0]-1),
                       max(col-r,0):min(col+r+1,scores.shape[1]-1)]
        scores[max(row - r, 0):min(row + r + 1, scores.shape[0] - 1),
        max(col - r, 0):min(col + r + 1, scores.shape[1] - 1)] = np.zeros(patch.shape)

    return np.asarray(keypoints)

