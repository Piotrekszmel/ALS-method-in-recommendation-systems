import numpy as np
import pandas as pd
from utils import recommendation, create_matrix, add_rows
import operator

LR = 0.1
D =  3
N = 300000
L = 10
DATA_PATH = "dataset/amazon-meta.txt"

df = pd.DataFrame(columns=['Group', 'Title', 'Id', 'Ratings'])
df = add_rows(df, DATA_PATH, N, "Book")
users = set(df["Id"])
df["index"] = range(0, len(df))

df.to_csv()
"""
R, U, P, User_id, Product_id, df = create_matrix(DATA_PATH, 10000, 3, ["Book", " DVD", "Video", "Music"])

df.to_csv(index=False)


R, U, P, User_id, Product_id, df = recommendation(DATA_PATH, D, N, L, LR, ["Book", " DVD", "Video", "Music"])

R_ = np.array(np.matmul(U.T, P))

print("XDDDDDDDDDDDDDDDDDDDDDDDDD")
for val in R_:
    for v in val:
        print(int(v), end=" ")
    print()
print("XDDDDDDDDDDDDDDDDDDDDDDDDD")
print("\n")

for val in R:
    for v in val:
        print(v, end=" ")
    print()

print("\n")

for val in U:
    for v in val:
        print(v, end=" ")
    print()

print("\n")

for val in P:
    for v in val:
        print(v, end=" ")
    print()
"""