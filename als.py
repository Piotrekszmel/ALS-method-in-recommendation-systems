from utils import create_matrix, create_R_up, objective_function
from gauss import matrix
import numpy as np

data_path = "dataset/amazon-meta.txt"
R, U, P, User_id, Product_id, df = create_matrix(data_path, 5, 3)

R_up = create_R_up(R)
U = np.ones((U.shape[0], U.shape[1]))
P = np.ones((P.shape[0], P.shape[1]))

print(objective_function(R_up, U, P, 0.1))
