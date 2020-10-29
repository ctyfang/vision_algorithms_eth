import numpy as np

def disparityToPointCloud(disp_img, K, baseline, left_img):
    # % points should be 3xN and intensities 1xN, where N is the amount of pixels
    # % which have a valid disparity. I.e., only return points and intensities
    # % for pixels of left_img which have a valid disparity estimate! The i-th
    # % intensity should correspond to the i-th point.

    nonzero_idxs = np.argwhere(disp_img != 0)
    N = nonzero_idxs.shape[0]
    points = np.zeros((N, 3))
    intensities = np.zeros(N)
    b = np.zeros((3, 1))
    b[0, 0] = baseline
    K_inv = np.linalg.inv(K)

    for point_idx in range(nonzero_idxs.shape[0]):

        row, col = nonzero_idxs[point_idx, :]
        intensities[point_idx] = left_img[row, col]

        p_0_vec = np.asarray([col, row, 1]).reshape((3, 1))
        p_1_vec = np.asarray([col-disp_img[row, col], row, 1]).reshape((3, 1))
        p_0_vec_hat = K_inv @ p_0_vec
        p_1_vec_hat = K_inv @ p_1_vec

        A = np.ones((3, 2))
        A[:, 0] = p_0_vec_hat.reshape((3, ))
        A[:, 1] = -p_1_vec_hat.reshape((3, ))

        # Solve for lambda
        lambdas, res, rank, s = np.linalg.lstsq(A.T @ A, A.T @ b)
        P = lambdas[0]*K_inv @ p_0_vec
        points[point_idx, :] = P.reshape((3,))

    return points, intensities
