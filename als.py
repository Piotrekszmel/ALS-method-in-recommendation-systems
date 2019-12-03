from data_loader import create_matrix
from gauss import matrix


data_path = "dataset/amazon-meta.txt"
R, U, P, User_id, Product_id, df = create_matrix(data_path, 5, 3)

"""
for i in range(len(U[1])):
    for j in range(len(P[1])):
        print(R[i][j], end=" ")
    print()
"""


print(R.shape)
print(U.shape)
print(P.shape)