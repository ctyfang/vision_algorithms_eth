import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

import copy
import time
import os

def estimatePoseDLT(p, P, K):
    """Given n 2D-3D correspondences, solve for the projection matrix
    (for normalized coordinates) M = [R|t]. p should be Nx2
    P should be Nx3 matrices."""

    N = p.shape[0]
    Q = np.zeros((2*N, 12))
    K_inv = np.linalg.inv(K)

    ones = np.ones((p.shape[0], 1))
    p_homo = np.concatenate([p, ones], axis=1)
    p_norm_homo = np.transpose(np.matmul(K_inv, np.transpose(p_homo)))
    P_homo = np.concatenate([P, ones], axis=1)

    # Formulate Q
    for correspondenceIndex in range(N):
        p_norm_i = p_norm_homo[correspondenceIndex, :]
        P_i = P_homo[correspondenceIndex, :]

        # Equation for u
        Q[2*correspondenceIndex, :4] = np.transpose(P_i)
        Q[2*correspondenceIndex, 8:] = np.transpose(-p_norm_i[0]*P_i)

        # Equation for v
        Q[2*correspondenceIndex+1, 4:8] = np.transpose(P_i)
        Q[2*correspondenceIndex+1, 8:] = np.transpose(-p_norm_i[1]*P_i)

    # Solve for M
    u, s, vh = np.linalg.svd(Q)
    M = np.transpose(vh)[:, -1].reshape((3, 4))

    # Correct M for scaling
    if M[2, 3] < 0:
        M *= -1
    R = M[:3, :3]
    u, s, vh = np.linalg.svd(R)
    R_proper = np.matmul(u, vh)
    alpha = np.linalg.norm(R_proper)/np.linalg.norm(R)
    M[:3, :3] = R_proper
    M[:3, 3] *= alpha
    return M


def reprojectPoints(P, M, K):
    """Given 3D points, P (Nx3), projection matrix for normalized coordinates,
    M (3x4), and the calibration matrix, K (3x3), compute the projected
    pixel coordinates for the points, p_reprojected (Nx2)"""
    ones = np.ones((P.shape[0], 1))
    P_homo = np.concatenate([P, ones], axis=1)
    P_C = np.transpose(np.matmul(M, np.transpose(P_homo)))
    P_C_normalized = P_C / P_C[:, 2].reshape((-1, 1))
    p_reprojected = np.transpose(np.matmul(K, np.transpose(P_C_normalized)))
    return p_reprojected


def drawPoints(image, pixels, color=(255, 0, 0), rectangle=False, rect_width=8):
    """Given an image and a set of pixels (Nx2), draw the pixels
    onto the image as circles and return the edited image."""
    new_image = copy.deepcopy(image)
    for pixel_index in range(pixels.shape[0]):
        pixel = pixels[pixel_index, :].astype(np.int)

        if rectangle:
            cv.rectangle(new_image, (pixel[0]-rect_width//2, pixel[1]-rect_width//2),
                         (pixel[0]+rect_width//2, pixel[1]+rect_width//2),
                         color, 2)
        else:
            cv.circle(new_image, (pixel[0], pixel[1]),
                      2, color, -1)

    return new_image

# Load data
K = np.genfromtxt("./data/K.txt").reshape((3, 3))
detected_corners = np.genfromtxt("./data/detected_corners.txt")
p_W_corners = np.genfromtxt("./data/p_W_corners.txt", delimiter=',')
image_index = 1
image_dir = "./data/images_undistorted"
num_images = len(os.listdir(image_dir))

# Single shot
image = cv.imread(os.path.join(image_dir, 'img_' + str(image_index).zfill(4) + '.jpg'))
corners_2D = detected_corners[image_index, :].reshape((-1, 2))
corners_3D = p_W_corners
M = estimatePoseDLT(corners_2D, corners_3D, K)
corners_2D_estimated = reprojectPoints(corners_3D, M, K)
image_gt = drawPoints(image, corners_2D, (0, 255, 0))
image_est = drawPoints(image_gt, corners_2D_estimated, (255, 0, 0), rectangle=True)
cv.imshow("G:detected, B:reprojected", image_est)
cv.waitKey(0)

plt.ion()
def plotTrajectory3D(framerate):
    sleep_time = round(1000./framerate)

    fig = plt.figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    x_arr, y_arr, z_arr = [], [], []
    for image_index in range(num_images):
        corners_2D = detected_corners[image_index, :].reshape((-1, 2))
        corners_3D = p_W_corners
        M = estimatePoseDLT(corners_2D, corners_3D, K)

        T = M[:, 3]
        R = M[:3, :3]
        R_inv = np.transpose(R)
        T_new = -R_inv @ T

        x_arr.append(T_new[0])
        y_arr.append(T_new[1])
        z_arr.append(T_new[2])
        u, v, w = R_inv[:, 0], R_inv[:, 1], R_inv[:, 2]
        plt.cla()
        ax.plot(x_arr, y_arr, z_arr, 'b')
        ax.quiver(x_arr[-1], y_arr[-1], z_arr[-1],
                  u, v, w, length=0.3, normalize=True)

        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        cv.imshow("Trajectory", image)
        cv.waitKey(sleep_time)

plotTrajectory3D(10)



