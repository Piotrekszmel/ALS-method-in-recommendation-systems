from utils import create_matrix, create_R_up, objective_function
from gauss import matrix
import numpy as np

lr = 0.1
d = 3
data_path = "dataset/amazon-meta.txt"

R, U, P, User_id, Product_id, df = create_matrix(data_path, 2, d)

R_up = create_R_up(R)
U = np.random.uniform(0, 1, (U.shape[0], U.shape[1]))
P = np.random.uniform(0, 1, (P.shape[0], P.shape[1]))

U = np.matrix(U)
P = np.matrix(P)
R = np.matrix(R)
"""
for i in range(U.shape[1]):
    I = [int(p) for r, u, p in R_up if u == i]
    A = np.dot(P[:, I], P[:, I].T) + lr * np.eye(d)
    V = np.sum([R[i, j] * P[:, j] for j in I], axis=0)
    G = matrix(A, V)
    U[:, i] = G.Gauss()
    print(U[:, i])
"""
A = np.asmatrix([[2, 4, 2, 0], [1, 0, -1, 1], [0, 1, 3, -1], [2, 1, 2, 1]], dtype=np.float64)
print(A.shape)
B = np.asmatrix([4, 2, 0, 6], dtype=np.float64).T
print(B.shape)

G = matrix(A, B)
print(G.Gauss())
#objective_function(R_up, U, P, lr)
