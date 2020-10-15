from copy import deepcopy
import numpy as np

def describeKeypoints(img, keypoints, r):
    """% Returns a Nx(2r+1)^2 matrix of image patch vectors based on image
        % img and a Nx2 matrix containing the keypoint coordinates.
        % r is the patch "radius"."""

    img = deepcopy(img)
    img = np.pad(img, r)

    N = keypoints.shape[0]
    descriptors = []

    for i, kp in enumerate(keypoints):
        row, col = kp
        # Account for the image padding
        row += r
        col += r

        patch = img[row-r:row+r+1, col-r:col+r+1]
        descriptors.append(np.reshape(patch, ((2*r+1)**2,)).tolist())

    return np.asarray(descriptors)