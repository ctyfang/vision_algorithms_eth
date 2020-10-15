import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from shi_tomasi import shi_tomasi
from harris import harris
from selectKeypoints import selectKeypoints
from describeKeypoints import describeKeypoints
from matchDescriptors import matchDescriptors
from plotMatches import plotMatches

""" Randomly chosen parameters that seem to work well - can you find better ones? """
corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 200
nonmaximum_supression_radius = 8
descriptor_radius = 9
match_lambda = 4

img = cv.imread('../data/000000.png')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

""" Part 1 - Calculate Corner Response Functions """

# Shi - Tomasi
shi_tomasi_scores = shi_tomasi(img, corner_patch_size)
# assert (min(len(shi_tomasi_scores) == len(shi_tomasi_scores)))

# % Harris
harris_scores = harris(img, corner_patch_size, harris_kappa)
# assert (min(size(harris_scores) == size(harris_scores)))

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img)

plt.subplot(2, 2, 2)
plt.imshow(img)

plt.subplot(2, 2, 3)
ax = plt.gca()
ax.imshow(shi_tomasi_scores)
plt.title('Shi-Tomasi Scores')

plt.subplot(2, 2, 4)
ax = plt.gca()
ax.imshow(harris_scores)
plt.title('Harris Scores')
# daspect([1 1 1])
plt.show()

"""Part 2 - Select keypoints"""
keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)
plt.figure()
plt.imshow(img)
plt.plot(keypoints[:, 1], keypoints[:, 0], 'rx')
plt.show()

"""Part 3 - Describe Keypoints, show 16 strongest descriptors"""
descriptors = describeKeypoints(img_gray, keypoints, descriptor_radius)
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    ax = plt.gca()
    patch_size = 2 * descriptor_radius + 1
    ax.imshow(np.reshape(descriptors[i, :], (patch_size, patch_size)).astype(np.uint8))
plt.show()

"""Part 4 - Match descriptors between the first two images"""
img_2 = cv.imread('../data/000001.png')
img_2_gray = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
harris_scores_2 = harris(img_2, corner_patch_size, harris_kappa)
keypoints_2 = selectKeypoints(harris_scores_2, num_keypoints, nonmaximum_supression_radius)
descriptors_2 = describeKeypoints(img_2_gray, keypoints_2, descriptor_radius)

matches = matchDescriptors(descriptors_2, descriptors, match_lambda)

plt.figure()
plt.imshow(img_2)
plt.plot(keypoints_2[:, 1], keypoints_2[:, 0], 'rx')
plotMatches(matches, keypoints_2, keypoints)
plt.show()

"""Part 5 - Match descriptors between all images"""
prev_kp, prev_desc = None, None
for img_idx in range(200):
    img = cv.imread('../data/' + str(img_idx).zfill(6) + '.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fig = plt.figure()
    canvas = FigureCanvasAgg(fig)
    plt.imshow(img)

    scores = harris(img, corner_patch_size, harris_kappa)
    kp = selectKeypoints(scores, num_keypoints, nonmaximum_supression_radius)
    plt.plot(kp[:, 1], kp[:, 0], 'rx')
    desc = describeKeypoints(img_gray, kp, descriptor_radius)

    if not isinstance(prev_kp, type(None)):
        matches = matchDescriptors(desc, prev_desc, match_lambda)
        plotMatches(matches, kp, prev_kp)

    canvas.draw()
    buf = canvas.buffer_rgba()
    X = np.asarray(buf)
    X = cv.cvtColor(X, cv.COLOR_RGB2BGR)
    cv.imshow("Matched Results", X)

    prev_kp = kp
    prev_desc = desc
    cv.waitKey(10)