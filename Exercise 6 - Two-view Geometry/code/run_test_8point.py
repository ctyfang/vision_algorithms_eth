import numpy as np
from eightpoint.fundamentalEightPoint import fundamentalEightPoint
from eightpoint.fundamentalEightPoint_normalized import fundamentalEightPoint_normalized
from eightpoint.distPoint2EpipolarLine import distPoint2EpipolarLine
N = 40
X = np.random.rand(N, 4)
X[:, 2] = X[:, 2]*5 + 10;

P1 = np.array([[500, 0, 320, 0],
               [0,   500, 240, 0],
               [0,   0,     1, 0]])

P2 = np.array([[500, 0, 320, -100],
               [0,   500, 240, 0],
               [0,   0,     1, 0]])

x1 = (P1 @ X.T).T # Image (ie projected) points
x2 = (P2 @ X.T).T
# x1 /= x1[:, 2].reshape((-1, 1))
# x2 /= x2[:, 2].reshape((-1, 1))
sigma = 1e-1
noisy_x1 = x1 + sigma * np.random.rand(x1.shape[0], x1.shape[1])
noisy_x2 = x2 + sigma * np.random.rand(x1.shape[0], x1.shape[1])

"""Fundamental matrix estimation via the 8-point algorithm"""

# Estimate fundamental matrix
# Call the 8-point algorithm on inputs x1,x2
F = fundamentalEightPoint(x1,x2);

# Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
cost_algebraic = np.linalg.norm(np.sum(np.multiply(x2, (F @ x1.T).T), axis=1), axis=0)/np.sqrt(N)
cost_dist_epi_line = distPoint2EpipolarLine(F,x1,x2)

print('Noise-free correspondences\n')
print(f'Algebraic error: {cost_algebraic}\n')
print(f'Geometric error: {cost_dist_epi_line} px\n\n')

"""Test with noise"""

# Estimate fundamental matrix
# Call the 8-point algorithm on noisy inputs x1,x2
F = fundamentalEightPoint(noisy_x1, noisy_x2);

# Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
cost_algebraic = np.linalg.norm(np.sum(np.multiply(noisy_x2, (F @ noisy_x1.T).T), axis=1), axis=0)/np.sqrt(N)
cost_dist_epi_line = distPoint2EpipolarLine(F,noisy_x1,noisy_x2);

print(f'Noisy correspondences (sigma={sigma}), with fundamentalEightPoint\n');
print(f'Algebraic error: {cost_algebraic}\n')
print(f'Geometric error: {cost_dist_epi_line} px\n\n')

"""Normalized 8-point algorithm"""
# Call the normalized 8-point algorithm on inputs x1,x2
Fn = fundamentalEightPoint_normalized(noisy_x1,noisy_x2);

# Check the epipolar constraint x2(i).' * F * x1(i) = 0 for all points i.
cost_algebraic = np.linalg.norm(np.sum(np.multiply(noisy_x2, (Fn @ noisy_x1.T).T), axis=1), axis=0)/np.sqrt(N)
cost_dist_epi_line = distPoint2EpipolarLine(Fn,noisy_x1,noisy_x2);

print(f'Noisy correspondences (sigma={sigma}), with fundamentalEightPoint\n');
print(f'Algebraic error: {cost_algebraic}\n')
print(f'Geometric error: {cost_dist_epi_line} px\n\n')
