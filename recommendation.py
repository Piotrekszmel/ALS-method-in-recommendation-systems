import numpy as np
import pandas as pd
from utils import recommendation, create_matrix, add_rows, create_matrices
import operator

LR = 0.4
D =  3
N = 497800
L = 10
DATA_PATH = "dataset/amazon-meta.txt"

data = pd.read_csv("amazon_data.csv", index_col=False)

title1 = data.loc[data["Title"] == "Make a Wish (Holiday Greetings Cards)"]
title2 = data.loc[data["Title"] == "The Man Who Sold the World"]
title3 = data.loc[data["Title"] == "The Three Stooges - Spook Louder"]
title4 = data.loc[data["Title"] == "True Grits : The Southern Foods Mail-Order Catalog "]
title5 = data.loc[data["Title"] == "Quick Toning: Thighs of Steel "]
title6 = data.loc[data["Title"] == "Simply Red - Live in London"]
title7 = data.loc[data["Title"] == "The Kama Sutra of Vatsyayana"]

idxs = {}
titles = [title1, title2, title5, title6, title7]


R, U, P = create_matrices(titles, D, 10)

R, U, P = recommendation(R, U, P, L, LR, D)
R = np.array(R)

R_ = np.array(np.matmul(U.T, P))

for val in R_:
    for v in val:
        print(int(v), end=" ")
    print()
print("\n\n")

for val in R:
    for v in val:
        print(int(v), end=" ")
    print()

print("\n")





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