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
N = 1000000
L = 30
DATA_PATH = "dataset/amazon-meta.txt"

LR = [0.5]
D = [1,2,3,4,5,6,7,8]  
indexes = []
reviews = []
rows = []
values = []
k = 0
#fig, ax1 = plt.subplots()

#ax2 = ax1.twinx()
#loops = [10, 30]
#ax1_colors = ["tab:green", "tab:red", "tab:brown"]
ax2_colors = ["tab:purple", "tab:orange", "tab:pink", "tab:green", "tab:red", "tab:brown", "tab:olive", "tab:blue"]


times = []
costs = []
values = []
diffs = np.zeros(shape=(len(LR), len(D)))
for y, dim in enumerate(D):
    for x, learning_rate in enumerate(LR):
        diff = 0 
        
        start = time.time()
        R, U, P = create_matrices("Book", dim,  100, 10, "amazon_data.csv")
        
        R_t = copy.deepcopy(R)

        for i in range(R.shape[0]):
            index = randrange(0, 9)
            if R[i][index] != 0.0:
                rows.append(i)
                indexes.append(index)
                reviews.append(R[i][index])
                R[i][index] = 0.0

        
        R, U, P, cost = recommendation(R, U, P, L, learning_rate, dim, 0)
        end = time.time()
        R = np.array(R)
        
        values.append(cost)
        costs.append(cost[-1])
        
        R_ = np.array(np.matmul(U.T, P))
        
        U = np.array(U)
        P = np.array(P)
        
        for i in range(len(indexes)):
                diff += abs(R_t[rows[i], indexes[i]] - int(R_[rows[i], indexes[i]]))
        print("cost = ", cost[-1])      
        #print("diff = ",diff)
        #print()
        diffs[x][y] = float(diff)
        times.append(end - start)
        #plt.plot([t for t in range(L)], [c for c in cost], label="d = " + str(dim), color=ax2_colors[k]) 
        #k += 1       
        #ax1.plot([dim for dim in D], [t for t in times], label="time for L = " + str(loops[w]), color=ax1_colors[w]) 

#ax1.legend(loc="upper right")
#ax2.legend(loc="upper left")
#ax1.set_xlabel("d")
for k in range(len(D)):
    plt.plot([t for t in range(L)], [c for c in values[k]], label="d = " + str(D[k]), color=ax2_colors[k])
plt.xlabel("iteration")
plt.ylabel("objective function value")
plt.legend()

plt.show()
#plt.savefig("100_10_L_C")
"""  
for i in range(0, len(D)):
    plt.plot([dim for dim in D], [diff for diff in diffs[i]], label="LR = " + str(LR[i]))

k = 0
for i in range(0, len(values)):
    if i % 3 == 0 and i != 0:
        k += 1
    plt.plot([iteration for iteration in range(L)], [c for c in values[i]], label="LR = " + str(LR[k]) + "D = " + str(D[k]))
plt.xlabel("Iteration")
plt.ylabel("Objective Function value")
plt.legend()
plt.savefig("100_10_cost")
plt.show()
"""

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
N = 1000000
L = 100
DATA_PATH = "dataset/amazon-meta.txt"

LR = [0.5]
D = [1,2,3,4,5,6,7,8]  
indexes = []
reviews = []
rows = []
values = []

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
loops = [10, 30]
ax1_colors = ["tab:green", "tab:red", "tab:brown"]
ax2_colors = ["tab:purple", "tab:orange", "tab:pink"]

for w, l in enumerate(loops):
    times = []
    costs = []
    values = []
    diffs = np.zeros(shape=(len(LR), len(D)))
    for y, dim in enumerate(D):
        for x, learning_rate in enumerate(LR):
            diff = 0 
            
            start = time.time()
            R, U, P = create_matrices("Book", dim,  300, 1010, "amazon_data.csv")
            
            R_t = copy.deepcopy(R)

            for i in range(R.shape[0]):
                index = randrange(0, 9)
                if R[i][index] != 0.0:
                    rows.append(i)
                    indexes.append(index)
                    reviews.append(R[i][index])
                    R[i][index] = 0.0

            
            R, U, P, cost = recommendation(R, U, P, l, learning_rate, dim, 0)
            end = time.time()
            R = np.array(R)
            
            values.append(cost)
            costs.append(cost[-1])
            
            R_ = np.array(np.matmul(U.T, P))
            
            U = np.array(U)
            P = np.array(P)
            
            for i in range(len(indexes)):
                    diff += abs(R_t[rows[i], indexes[i]] - int(R_[rows[i], indexes[i]]))
            print("cost = ", cost[-1])      
            #print("diff = ",diff)
            #print()
            diffs[x][y] = float(diff)
            times.append(end - start)
    ax1.plot([dim for dim in D], [t for t in times], label="time for L = " + str(loops[w]), color=ax1_colors[w])       
    ax2.plot([dim for dim in D], [c for c in costs], label="objective function value for L = " + str(loops[w]), color=ax2_colors[w])        


ax1.legend(loc="upper right")
ax2.legend(loc="upper left")
ax1.set_xlabel("d")

plt.show()
plt.savefig("500_1100_3.3")
  
for i in range(0, len(D)):
    plt.plot([dim for dim in D], [diff for diff in diffs[i]], label="LR = " + str(LR[i]))

k = 0
for i in range(0, len(values)):
    if i % 3 == 0 and i != 0:
        k += 1
    plt.plot([iteration for iteration in range(L)], [c for c in values[i]], label="LR = " + str(LR[k]) + "D = " + str(D[k]))
plt.xlabel("Iteration")
plt.ylabel("Objective Function value")
plt.legend()
plt.savefig("100_10_cost")
plt.show()
"""

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
