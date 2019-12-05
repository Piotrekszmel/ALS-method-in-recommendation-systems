from utils import recommendation

LR = 0.2
D =  3
N = 2
L = 100
DATA_PATH = "dataset/amazon-meta.txt"


R, U, P, User_id, Product_id, df = recommendation(DATA_PATH, D, N, L, LR)


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