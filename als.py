from utils import create_matrix, create_R_up
from gauss import matrix


data_path = "dataset/amazon-meta.txt"
R, U, P, User_id, Product_id, df = create_matrix(data_path, 5, 3)

R_up = create_R_up(R)

