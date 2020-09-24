import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

poses = np.genfromtxt('data/poses.txt')
D = np.genfromtxt('data/D.txt')
K = np.genfromtxt('data/K.txt')
rows, cols = [6, 9]
squareSize = 0.04

def poseVectorToTransformationMatrix(pose):
    R, _ = cv.Rodrigues(pose[:3])
    T = pose[3:].reshape((3,))
    transformation_matrix = np.zeros((3, 4))
    transformation_matrix[:, :3] = R
    transformation_matrix[:, 3] = T
    return transformation_matrix


def distortPoint(point):
    """Given normalized image coordinate (x, y), apply distortion coeffs
    return homogeneous point (x', y', 1)"""
    r = np.sqrt(point[0] ** 2 + point[1] ** 2)
    coeff = (1 + D[0] * (r ** 2) + D[1] * (r ** 4))
    point[0] *= coeff
    point[1] *= coeff
    return point


def projectPoints(points, transformation_matrix=None, distort=False):
    """Take in a Nx3 matrix of 3D points, return an Nx2 matrix of 2D points
        by applying the given transformation matrix. Distort the points
        using global variable distortion coefficients."""
    projectedPoints = []
    for point in points:
        point = np.asarray(point).reshape((3, 1))
        if type(transformation_matrix) is not type(None):
            R = transformation_matrix[:, :3]
            T = transformation_matrix[:, 3].reshape((3, 1))
            point = np.matmul(R, point)
            point += T
            point /= point[2]

        if distort:
            point = distortPoint(point)

        projectedPoint = np.matmul(K, point)
        projectedPoint /= projectedPoint[2]
        projectedPoints.append(projectedPoint[:2])
    return np.asarray(projectedPoints).reshape((-1, 2))


# 2.2 Writing and testing the projection function
img = cv.imread('data/images_undistorted/img_0001.jpg')
pose = poses[0, :]
transformation_matrix = poseVectorToTransformationMatrix(pose)

objectPoints = []
for row in range(rows):
    for col in range(cols):
        objectPoints.append([col*squareSize, row*squareSize, 0])
objectPoints = np.asarray(objectPoints)
projectedPoints = projectPoints(objectPoints, transformation_matrix)

plt.imshow(img)
plt.scatter(projectedPoints[:, 0], projectedPoints[:, 1])
plt.show()


# 2.3 Drawing the Cube
def generateCubePoints(cubeSideLength=0.08):
    """layer by layer, top-left, top-right, bottom-right, bottom-left"""
    objectPoints = []
    objectPoints.append([0., 0., 0.])
    objectPoints.append([cubeSideLength, 0., 0.])
    objectPoints.append([cubeSideLength, cubeSideLength, 0.])
    objectPoints.append([0., cubeSideLength, 0.])

    objectPoints.append([0., 0., -cubeSideLength])
    objectPoints.append([cubeSideLength, 0., -cubeSideLength])
    objectPoints.append([cubeSideLength, cubeSideLength, -cubeSideLength])
    objectPoints.append([0., cubeSideLength, -cubeSideLength])

    return np.asarray(objectPoints).reshape((-1, 3))

def drawCubePoints(img, cubeProjectedPoints):
    plt.imshow(img)
    plt.plot(cubeProjectedPoints[[0, 1], 0], cubeProjectedPoints[[0, 1], 1])
    plt.plot(cubeProjectedPoints[[1, 2], 0], cubeProjectedPoints[[1, 2], 1])
    plt.plot(cubeProjectedPoints[[2, 3], 0], cubeProjectedPoints[[2, 3], 1])
    plt.plot(cubeProjectedPoints[[3, 0], 0], cubeProjectedPoints[[3, 0], 1])

    plt.plot(cubeProjectedPoints[[4, 5], 0], cubeProjectedPoints[[4, 5], 1])
    plt.plot(cubeProjectedPoints[[5, 6], 0], cubeProjectedPoints[[5, 6], 1])
    plt.plot(cubeProjectedPoints[[6, 7], 0], cubeProjectedPoints[[6, 7], 1])
    plt.plot(cubeProjectedPoints[[7, 4], 0], cubeProjectedPoints[[7, 4], 1])

    plt.plot(cubeProjectedPoints[[0, 4], 0], cubeProjectedPoints[[0, 4], 1])
    plt.plot(cubeProjectedPoints[[1, 5], 0], cubeProjectedPoints[[1, 5], 1])
    plt.plot(cubeProjectedPoints[[2, 6], 0], cubeProjectedPoints[[2, 6], 1])
    plt.plot(cubeProjectedPoints[[3, 7], 0], cubeProjectedPoints[[3, 7], 1])
    plt.show()

cubeSideLength = 0.12
cubeObjectPoints = generateCubePoints(cubeSideLength)
cubeProjectedPoints = projectPoints(cubeObjectPoints, transformation_matrix)
drawCubePoints(img, cubeProjectedPoints)


# 3.1 Accounting for lens distortion
img = cv.imread('data/images/img_0001.jpg')
dProjectedPoints = projectPoints(objectPoints, transformation_matrix, distort=True)
plt.figure()
plt.imshow(img)
plt.scatter(dProjectedPoints[:, 0], dProjectedPoints[:, 1])
plt.show()

# 3.2 Undistorting Images
import copy
img_undistorted = copy.copy(img)
K_inv = np.linalg.inv(K)

## Vectorized
# grid_x, grid_y = np.meshgrid(range(img_undistorted.shape[1]), range(img_undistorted.shape[0]))
# grid_x = grid_x.flatten().reshape((-1, 1))
# grid_y = grid_y.flatten().reshape((-1, 1))
# grid_ones = np.ones(grid_x.shape)
# points = np.concatenate([grid_x, grid_y, grid_ones], axis=1)
# points = np.transpose(np.matmul(K_inv, np.transpose(points)))
# points /= points[:, 2].reshape((-1, 1))
# dPoints = copy.copy(points)
# for point_idx in range(points.shape[0]):
#     point = points[point_idx, :]
#     dPoints[point_idx, :] = distortPoint(point)
# dPoints = np.transpose(np.matmul(K, np.transpose(dPoints))).astype(np.int)
# img_undistorted = img[dPoints[:, 1], dPoints[:, 0]]

## Brute Force
for row in range(img_undistorted.shape[0]):
    for col in range(img_undistorted.shape[1]):
        point = np.asarray([col, row, 1]).reshape((3, 1))
        point = np.matmul(K_inv, point)
        point /= point[2]

        dPoint = distortPoint(point)
        dProjectedPoint = np.matmul(K, dPoint)
        dProjectedPoint /= dProjectedPoint[2]
        img_undistorted[row, col] = img[int(dProjectedPoint[1]), int(dProjectedPoint[0])]

plt.imshow(img_undistorted)
plt.show()




