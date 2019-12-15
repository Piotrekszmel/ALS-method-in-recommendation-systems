import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import copy
from random import randrange

from utils import recommendation, create_matrix, add_rows, create_matrices


LR = 0.2
D = 3
N = 1000000
L = 100
LR = [0.1, 0.2, 0.3]
D = [8, 10, 12]  
DATA_PATH = "dataset/amazon-meta.txt"

values = []
indexes = []
reviews = []
rows = []
diffs = np.zeros(shape=(len(D), len(LR)))

for y, dim in enumerate(D):
    for x, learning_rate in enumerate(LR):
        diff = 0 
        
        R, U, P = create_matrices("Book", dim,  300, 1100, "amazon_data.csv")
        R_t = copy.deepcopy(R)

        for i in range(R.shape[0]):
            index = randrange(0, 9)
            if R[i][index] != 0.0:
                rows.append(i)
                indexes.append(index)
                reviews.append(R[i][index])
                R[i][index] = 0.0

        R, U, P, cost = recommendation(R, U, P, L, learning_rate, dim, 0)
        R = np.array(R)
        
        values.append(cost)

        R_ = np.array(np.matmul(U.T, P))
        
        U = np.array(U)
        P = np.array(P)

        for i in range(len(indexes)):
                diff += abs(R_t[rows[i], indexes[i]] - int(R_[rows[i], indexes[i]]))
        diffs[x][y] = float(diff)
 
for i in range(0, len(D)):
    plt.plot([dim for dim in D], [diff for diff in diffs[i]], label="LR = " + str(LR[i]))
plt.xlabel("d")
plt.ylabel("absolute difference between R and R~")
plt.legend()
plt.savefig("1200_1100")
plt.show()




""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import copy
from random import randrange

from utils import recommendation, create_matrix, add_rows, create_matrices


LR = 0.2
D = 3
N = 1000000
L = 100
DATA_PATH = "dataset/amazon-meta.txt"

LR = [0.1, 0.2, 0.3]
D = [8, 10, 12]  
indexes = []
reviews = []
rows = []
diffs = np.zeros(shape=(len(D), len(LR)))
values = []

for y, dim in enumerate(D):
    for x, learning_rate in enumerate(LR):
        diff = 0 
        
        R, U, P = create_matrices("Book", dim,  1500, 1200, "amazon_data.csv")
        
        R_t = copy.deepcopy(R)

        for i in range(R.shape[0]):
            index = randrange(0, 9)
            if R[i][index] != 0.0:
                rows.append(i)
                indexes.append(index)
                reviews.append(R[i][index])
                R[i][index] = 0.0

        
        R, U, P, cost = recommendation(R, U, P, L, learning_rate, dim, 0)
        R = np.array(R)
        
        values.append(cost)

        R_ = np.array(np.matmul(U.T, P))
        
        U = np.array(U)
        P = np.array(P)
        
        for i in range(len(indexes)):
                diff += abs(R_t[rows[i], indexes[i]] - int(R_[rows[i], indexes[i]]))
                
        diffs[x][y] = float(diff)
        
        
        
    
for i in range(0, len(D)):
    plt.plot([dim for dim in D], [diff for diff in diffs[i]], label="LR = " + str(LR[i]))

for i in range(0, len(D)):
    plt.plot([iteration for iteration in range(L)], [c for c in values[i]], label="LR = " + str(LR[i]) + "D = " + str(D[i]))
plt.xlabel("Iteration")
plt.ylabel("Objective Function value")
plt.legend()
plt.savefig("100_10_cost")
plt.show()





"""
