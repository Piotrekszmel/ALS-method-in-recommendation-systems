import numpy as np
import pandas as pd
from utils import recommendation, create_matrix, add_rows, create_matrices
import operator

LR = 0.2
D = 3
N = 497800
L = 10
DATA_PATH = "dataset/amazon-meta.txt"

data = pd.read_csv("amazon_data.csv", index_col=False)

title1 = data.loc[data["Title"] == "Make a Wish (Holiday Greetings Cards)"]
title2 = data.loc[data["Title"] == "The Man Who Sold the World"]
title3 = data.loc[data["Title"] == "The Three Stooges - Spook Louder"]
title4 = data.loc[data["Title"] == "True Grits : The Southern Foods Mail-Order Catalog"]
title5 = data.loc[data["Title"] == "Quick Toning: Thighs of Steel"]
title6 = data.loc[data["Title"] == "Simply Red - Live in London"]
title7 = data.loc[data["Title"] == "The Kama Sutra of Vatsyayana"]

LR =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
learning_rates = [0.8]

for learning_rate in learning_rates:
    idxs = {}
    diff = 0
    titles = [title1, title2, title6, title5, title3, title4]
    

    R, U, P = create_matrices(titles, D, 10)

    R, U, P = recommendation(R, U, P, L, learning_rate, D, 0)
    R = np.array(R)

    R_ = np.array(np.matmul(U.T, P))
    
    U = np.array(U)
    P = np.array(P)


    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            diff += abs(R[i][j] - R_[i][j])
        
    print("\ndiff = ", diff, "\nlr = ", learning_rate)

    print("R = ", R.shape)
    print("R_ = ", R_.shape)
    print("U = ", U.shape)
    print("P = ", P.shape)
    print("\n")
    """
    for val in R_:
        for v in val:
            print(int(v), end=" ")
        print()
    print("\n")

    for val in U:
        for v in val:
            print(v, end=" ")
        print()
    print("\n")
    """
    for val in P:
        for v in val:
            print(v, end="   ")
        print()
    print("\n")
    """
    for val in R:
        for v in val:
            print(int(v), end=" ")
        print()
    print("\n")
    """

"""
lr = 1 50.76739855807079 3
lr = 0.4 51.85975202875956 3

"""



"""
for val in R:
    for v in val:
        print(int(v), end=" ")
    print()

print("\n")
"""




"""
df = pd.DataFrame(columns=['Group', 'Title', 'Id', 'Ratings'])
df = add_rows(df, DATA_PATH, N, "Book")
print("Indexing...")
users = set(df["Id"])
df["index"] = range(0, len(df))
print("\nSaving...")
df.to_csv("amazon_data.csv")

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