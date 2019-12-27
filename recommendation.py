import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import copy
import time
import math
from random import randrange

from utils import recommendation, create_matrix, add_rows, create_matrices, objective_function, average_error


L = 20
DATA_PATH = "dataset/amazon-meta.txt"

LR = [0.1, 0.2, 0.3]
D = [4, 5, 6, 7, 8]  

reviews = []
values = []
k = 0

ax2_colors = ["tab:purple", "tab:orange", "tab:pink", "tab:green", "tab:red", "tab:brown", "tab:olive", "tab:blue"]


times = []
costs = []
values = []



for x, learning_rate in enumerate(LR):
    costs = []
    for y, dim in enumerate(D):
        R_test = []
        indexes = []
        rows = []
        diff = 0 
        reviews = []
        start = time.time()
        R, U, P = create_matrices("Book", dim,  500,  200, "amazon_data.csv")
        
        R_t = copy.deepcopy(R)
       
        for i in range(R.shape[0]):
            if len(reviews) == 10:
                break
            index = randrange(0, 9)
            if R[i][index] != 0.0 and len(np.flatnonzero(R[i])) > 1:
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
        costs.append(average_error(np.array(R_test), U, P))
    plt.plot([d for d in D], [c for c in costs], label="lr = " + str(learning_rate))
        
        
plt.ylabel("error")
plt.legend()

plt.show()

