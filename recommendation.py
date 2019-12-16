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
