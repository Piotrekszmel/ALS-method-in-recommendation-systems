"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import copy
import time
import math
from random import randrange

from utils import recommendation, create_matrix, add_rows, create_matrices


LR = 0.2
D = 3
users = 10000
products = 1000
L = 30
DATA_PATH = "dataset/amazon-meta.txt"
R, U, P = create_matrices("Book", dim,  users, products, "amazon_data.csv")
R, U, P, cost = recommendation(R, U, P, L, LR, D, 0)

R = np.array(R)
R_ = np.array(np.matmul(U.T, P))
U = np.array(U)
P = np.array(P)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import copy
import time
import math
from random import randrange

from utils import recommendation, create_matrix, add_rows, create_matrices, objective_function


LR = 0.2
D = 3
N = 1000000
L = 20
DATA_PATH = "dataset/amazon-meta.txt"

LR = [0.2]
D = [9,10,11,12]  

reviews = []
values = []
k = 0

ax2_colors = ["tab:purple", "tab:orange", "tab:pink", "tab:green", "tab:red", "tab:brown", "tab:olive", "tab:blue"]


times = []
costs = []
values = []

for y, dim in enumerate(D):
    for x, learning_rate in enumerate(LR):
        R_test = []
        indexes = []
        rows = []
        diff = 0 
        
        start = time.time()
        R, U, P = create_matrices("Book", dim,  300,  1010, "amazon_data.csv")
        
        R_t = copy.deepcopy(R)
        
        for i in range(R.shape[0]):
            index = randrange(0, 1010)
            if R[i][index] != 0.0 and len(np.flatnonzero(R[i])) > 20:
                rows.append(i)
                indexes.append(index)
                reviews.append(R[i][index])
                R[i][index] = 0.0
        

        
        R, U, P, cost = recommendation(R, U, P, L, learning_rate, dim, 0)
        end = time.time()
        R = np.array(R)
        
        values.append(cost)
        
        R_ = np.array(np.matmul(U.T, P))
        
        U = np.array(U)
        P = np.array(P)
        
        for i, j in zip(rows, indexes):
            R_test.append([R_t[i][j], i, j])
        
        times.append(end - start)
         
        costs.append(objective_function(np.array(R_test), U, P, 0.2))
        print(y)
    
        
        
        
        

plt.plot([t for t in times], [c for c in costs], label="L = " + str(LR[0]))
plt.xlabel("d")
plt.ylabel("objective function value")
plt.legend()

plt.show()

plt.clf()
plt.plot([dim for dim in D], [t for t in times], label="time")
plt.legend()
plt.show()