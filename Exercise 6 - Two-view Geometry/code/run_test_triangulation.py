import numpy as np
from triangulation.linearTriangulation import linearTriangulation

N = 10 # Number of 3D Poits
P = np.random.randn(N, 4)
P[:, 2] = P[:, 3]*5 + 10
P[:, 3] = 1

# Test Linear Triangulation

M1 = np.array([[500, 0,  320, 0],
               [0  ,500, 240, 0],
               [0  ,0, 1, 0]])
M2 = np.array([[500, 0,  320, -100],
               [0  ,500, 240, 0],
               [0  ,0, 1, 0]])

p1 = (M1 @ P.T).T; # Image(i.e., projected) points
p2 = (M2 @ P.T).T;

P_est = linearTriangulation(p1, p2, M1, M2);

print(f'P_est-P=\n{P_est-P}')