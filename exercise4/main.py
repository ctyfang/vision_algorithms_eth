import cv2 as cv
import numpy as np
from scipy.ndimage import maximum_filter
from sklearn.neighbors import NearestNeighbors

from copy import deepcopy
import matplotlib.pyplot as plt

num_scales = 3                  # Scales per octave.
num_octaves = 5                 # Number of octaves.
sigma = 1.6                     # Initial sigma
contrast_threshold = 0.04       # Threshold for noise suppression
image_file_1 = './images/img_1.jpg'
image_file_2 = './images/img_2.jpg'
rescale_factor = 0.2            # Rescaling of the original image for speed.
max_ratio = 0.8
match_threshold = 100

images = [cv.imread(image_file_1),
          cv.imread(image_file_2)]

for i, image in enumerate(images):
    images[i] = cv.resize(image, (-1, -1), fx=rescale_factor, fy=rescale_factor)
    images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY)

kpt_locations = [[] for i in range(len(images))]
descriptors = [[] for i in range(len(images))]

for img_idx in [0, 1]:
#     % Write code to compute:
#     % 1)    image pyramid. Number of images in the pyarmid equals
#     %       'num_octaves'.
#     % 2)    blurred images for each octave. Each octave contains
#     %       'num_scales + 3' blurred images.
#     % 3)    'num_scales + 2' difference of Gaussians for each octave.
    img = deepcopy(images[img_idx]).astype(np.float64)
    blurred_imgs = []
    dogs = []
    for octave in range(num_octaves):
        current_img = cv.resize(img, (-1, -1),
                                fx=1/(2**octave), fy=1/(2**octave))
        current_dogs = []
        current_blurred_imgs = []
        for scale in range(-1, num_scales+2):
            current_sigma = 2**(scale/num_scales) * sigma
            current_blurred = cv.GaussianBlur(current_img, None,
                                              current_sigma, borderType=cv.BORDER_REFLECT)
            current_blurred_imgs.append(current_blurred)
            if scale >= 0:
                current_dogs.append(current_blurred_imgs[scale+1] -
                                    current_blurred_imgs[scale])

        blurred_imgs.append(current_blurred_imgs)
        dogs.append(np.asarray(current_dogs))

    # 4)    Compute the keypoints with non-maximum suppression and
    #       discard candidates with the contrast threshold.
    keypoint_volumes = []
    for octave_idx, dog in enumerate(dogs):
        dog = np.moveaxis(dog, 0, -1)
        H, W, D = dog.shape

        dog[dog < contrast_threshold] = 0
        dogs[octave_idx] = dog
        keypoint_volumes.append(np.zeros(dog.shape))

        for scale in range(1, D-1):
            for row in range(1, H-1):
                for col in range(1, W-1):

                    current_window = dog[row-1:row+2,
                                         col-1:col+2,
                                         scale-1:scale+2]
                    max_idx = np.argmax(current_window, axis=None)
                    max_idx = np.unravel_index(max_idx, current_window.shape)

                    if max_idx == (1, 1, 1):
                        keypoint_volumes[octave_idx][row, col, scale] = 1
                        dog[row - 1:row + 2,
                            col - 1:col + 2,
                            scale - 1:scale + 2] = 0    # NMS
        # max_filtered = maximum_filter(dog, (3, 3, 3))
        # max_locs = np.argwhere(np.equal(max_filtered, dog))
        # max_locs = max_locs[np.logical_or(max_locs[:, 2] != 0, ]
        keypoint_volumes[octave_idx] = keypoint_volumes[octave_idx][:, :, 1:D-1]
        print(f"Image {img_idx}, Octave {octave_idx}, number of keypoints : {np.sum(keypoint_volumes[octave_idx])}")

    # 5)    Given the blurred images and keypoints, compute the
    #       descriptors. Discard keypoints/descriptors that are too close
    #       to the boundary of the image. Hence, you will most likely
    #       lose some keypoints that you have computed earlier.
    for octave_idx, keypoint_volume in enumerate(keypoint_volumes):
        keypoint_locs = np.argwhere(keypoint_volume == 1)

        for keypoint_idx in range(keypoint_locs.shape[0]):
            row, col, diff_idx = keypoint_locs[keypoint_idx, :]
            max_row, max_col = img.shape[0]/(2**octave_idx), img.shape[1]/(2**octave_idx)
            if row < 7 or col < 7 or row > max_row-9 or col > max_col-9:
                continue
            kpt_locations[img_idx].append([col*(2**octave_idx), row*(2**octave_idx)])
            scale_idx = diff_idx + 1
            img = blurred_imgs[octave_idx][scale_idx]

            patch = img[row-7:row+9, col-7:col+9]
            dx = cv.Sobel(patch, cv.CV_64F, 1, 0, borderType=cv.BORDER_REFLECT)
            dy = cv.Sobel(patch, cv.CV_64F, 0, 1, borderType=cv.BORDER_REFLECT)
            mag = np.sqrt(dx**2 + dy**2)
            mag = cv.GaussianBlur(mag, None, 1.5*16, borderType=cv.BORDER_REFLECT)
            angle = np.arctan2(dy, dx)

            descriptor = []
            counts = 0
            for block_row in range(4):
                for block_col in range(4):
                    counts += 1
                    block_mag = mag[block_row*4:block_row*4+4,
                                    block_col*4:block_col*4+4]
                    block_angle = angle[block_row*4:block_row*4+4,
                                        block_col*4:block_col*4+4]
                    block_angle[block_angle < 0] += 2*np.pi
                    block_mag = block_mag.reshape((-1, 1))
                    block_angle = block_angle.reshape((-1, 1))
                    n, bins = np.histogram(block_angle, bins=np.linspace(0, 2*np.pi, 9),
                                           weights=block_mag)
                    descriptor += n.tolist()
            descriptor = np.asarray(descriptor)
            descriptor /= np.linalg.norm(descriptor)
            descriptors[img_idx].append(descriptor)

    kpt_locations[img_idx] = np.asarray(kpt_locations[img_idx])
    descriptors[img_idx] = np.asarray(descriptors[img_idx])

    # Visualize key points
    original_img = deepcopy(images[img_idx])
    for keypoint_idx in range(kpt_locations[img_idx].shape[0]):
        keypoint = kpt_locations[img_idx][keypoint_idx, :].tolist()
        center = tuple([keypoint[0], keypoint[1]]) # Keypoint is stored as row, col
        cv.circle(original_img, center, radius=3, color=(255, 0, 0), thickness=3)
    cv.imshow(f"Keypoints for image {img_idx}", original_img)
    cv.waitKey(0)

# Finally, match the descriptors using the function 'matchFeatures' and
# visualize the matches with the function 'showMatchedFeatures'.
# If you want, you can also implement the matching procedure yourself using
# 'knnsearch'.
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(descriptors[0])
dists, nbr_idxs = nbrs.kneighbors(descriptors[1])

good_matches = {}
for query_idx in range(descriptors[1].shape[0]):
    if dists[query_idx, 0]/dists[query_idx, 1] < max_ratio and dists[query_idx, 0] < match_threshold:
        good_matches[query_idx] = nbr_idxs[query_idx, 0]
print(f"Number of passable matches : {len(list(good_matches.keys()))}")

# Draw matches
concat_img = np.concatenate([images[0], images[1]], axis=1)
for (query_idx, train_idx) in good_matches.items():
    query_keypt = kpt_locations[1][query_idx, :]
    train_keypt = kpt_locations[0][train_idx, :]
    query_keypt[0] += images[0].shape[1]

    cv.circle(concat_img, tuple(train_keypt), 3, (255, 0, 0), -1)
    cv.circle(concat_img, tuple(query_keypt), 3, (0, 0, 255), -1)
    cv.line(concat_img, tuple(train_keypt), tuple(query_keypt), (255, 0, 0), thickness=2)
cv.imshow("Matches", concat_img)
cv.waitKey(0)