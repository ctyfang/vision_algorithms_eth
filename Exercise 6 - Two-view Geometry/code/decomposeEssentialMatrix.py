import numpy as np

def decomposeEssentialMatrix(E):
    """% Given an essential matrix, compute the camera motion, i.e.,  R and T such
    % that E ~ T_x R
    %
    % Input:
    %   - E(3,3) : Essential matrix
    %
    % Output:
    %   - R(3,3,2) : the two possible rotations
    %   - u3(3,1)   : a vector with the translation information"""

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    u, s, vh = np.linalg.svd(E)
    u3 = u[:, -1]

    R = np.zeros((2, 3, 3))
    R1 = u @ W @ vh
    R2 = u @ W.T @ vh

    if np.linalg.det(R1) < 0:
        R1 *= -1

    if np.linalg.det(R2) < 0:
        R2 *= -1

    R[0, :, :] = R1
    R[1, :, :] = R2

    return R, u3
