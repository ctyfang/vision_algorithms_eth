import numpy as np

from triangulation.linearTriangulation import linearTriangulation

def disambiguateRelativePose(Rots, u3, p1, p2, K0, K1):
    """% DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
    % four possible configurations) by returning the one that yields points
    % lying in front of the image plane (with positive depth).
    %
    % Arguments:
    %   Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
    %   u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
    %   p1   -  Nx3 homogeneous coordinates of point correspondences in image 1
    %   p2   -  Nx3 homogeneous coordinates of point correspondences in image 2
    %   K1   -  3x3 calibration matrix for camera 1
    %   K2   -  3x3 calibration matrix for camera 2
    %
    % Returns:
    %   R -  3x3 the correct rotation matrix
    %   T -  3x1 the correct translation vector
    %
    %   where [R|t] = T_C1_C0 = T_C1_W is a transformation that maps points
    %   from the world coordinate system (identical to the coordinate system of camera 0)
    %   to camera 1.
    %
    """

    R_arr = [Rots[0, :, :], Rots[1, :, :], Rots[0, :, :], Rots[1, :, :]]
    t_arr = [u3, u3, -u3, -u3]
    most_forward_points = 0
    best_cfg = None
    for config_idx in range(4):
        M1 = np.zeros((3, 4))
        M2 = np.zeros((3, 4))
        M1[:3, :3] = np.eye(3)
        M1 = K0 @ M1
        M2[:3, :3] = R_arr[config_idx]
        M2[:3, -1] = t_arr[config_idx].reshape((-1,))
        M2 = K1 @ M2

        P = linearTriangulation(p1, p2, M1, M2)
        num_forward_points = np.sum(P[:, 2] > 0)

        if num_forward_points > most_forward_points:
            most_forward_points = num_forward_points
            best_cfg = config_idx

    return R_arr[best_cfg], t_arr[best_cfg]
