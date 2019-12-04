from utils import create_matrix, create_R_up, objective_function
from gauss import matrix
import numpy as np

data_path = "dataset/amazon-meta.txt"
R, U, P, User_id, Product_id, df = create_matrix(data_path, 20, 3)

R_up = create_R_up(R)
U = np.ones((U.shape[0], U.shape[1]))
P = np.ones((P.shape[0], P.shape[1]))

for i in range(U.shape[1]):
    I = [(p, u) for r, u, p in R_up if u == i]
    print(I)
    #A = np.matmul(P)
    #U[:, i] = gauss()
#objective_function(R_up, U, P, 0.1)
