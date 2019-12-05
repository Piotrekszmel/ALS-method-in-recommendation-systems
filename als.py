from utils import create_matrix, create_R_up, objective_function
from gauss import matrix
import numpy as np

lr = 0.2
d =  5
n = 5
data_path = "dataset/amazon-meta.txt"

R, U, P, User_id, Product_id, df = create_matrix(data_path, n, d)

R_up = create_R_up(R)
U = np.random.uniform(0, 1, (U.shape[0], U.shape[1]))
P = np.random.uniform(0, 1, (P.shape[0], P.shape[1]))

U = np.matrix(U)
P = np.matrix(P)
R = np.matrix(R)
for i in range(10):
    for i in range(U.shape[1]):
        I = [int(p) for r, u, p in R_up if u == i]
        A = np.dot(P[:, I], P[:, I].T) + lr * np.eye(d)
        V = np.sum([R[i, j] * P[:, j] for j in I], axis=0)
        G = matrix(A, V)
        U[:, i] = G.Gauss()
        
    for i in range(P.shape[1]):
        I = [int(u) for r, u, p in R_up if u == i]
        B = np.dot(U[:, I], U[:, I].T) + lr * np.eye(d)
        
        V = np.sum([R[i, j] * U[:, j] for j in I], axis=0)
        G = matrix(B, V)
        P[:, i] = G.Gauss()
        
    value = objective_function(R_up, U, P, lr)
    print(value)
