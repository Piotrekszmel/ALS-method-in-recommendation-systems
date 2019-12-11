import pandas as pd
import numpy as np 
import operator

from gauss import matrix


def parse_text(size, data_path, category):
    text = open(data_path, "r")
    titles = []
    final_list = []
    rating = 0
    product_flag = 0
    title_flag = 0
    
    for row in text:
        row = row.replace("\n", "")
        
        if len(titles) == size:
            print("\nbreaking...")
            break
        
        if 'group: ' in row and any(cat in row for cat in category):
            product_type = "".join(row.split(':')[1]).strip()
            product_flag = 1
            
        if 'title: ' in row and product_flag == 1:
            row = row.split()
            title = " ".join(word for word in row[1:])
            title_flag = 1
            
        if 'rating:' in row and 'avg rating' not in row:
            row = row.split(":")
            rating = row[2][1].strip()
            customer_id = row[1][:-6].strip()
        
        if rating != 0 and product_flag == 1 and title_flag:
            final_list.append([product_type, title, customer_id, rating])
            titles.append(title)
            rating = 0
            product_flag = 0,
            if len(titles) % 100000 == 0:
                print(len(titles))
    return final_list


def add_rows(df, data_path, size, category):
    data = parse_text(size, data_path, category)
    print("appending...")
    for row in data:
        df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
    print('DataFrame Updated!')
    
    return df


def create_matrix(data_path, size, d, category):
    df = pd.DataFrame(columns=['Group', 'Title', 'Id', 'Ratings'])
    df = add_rows(df, data_path, size, category)
    users = set(df["Id"])
    df["index"] = range(0, len(df))
    
    User_id = {key: val for val, key in enumerate(users)}
    Product_id = {key : val for val, key in enumerate(set(df["Title"]))}

    R = np.zeros(shape=(len(users), len(Product_id)))
    U = np.zeros(shape=(d, len(users)))
    P = np.zeros(shape=(d, len(Product_id)))

    for _, row in df.iterrows():
        try:
            R[User_id[row["Id"]]][Product_id[row["Title"]]] = row["Ratings"]
        except:
            continue
        
    return np.asarray(R), np.asarray(U), np.asarray(P), User_id, Product_id, df


def create_R_up(R):
    R_up = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i][j] != 0:
                R_up.append([R[i][j], i, j])
    return np.asarray(R_up)


def objective_function(R_up, U, P, lr):
    value = 0
    for r, i, j in R_up:
        i = int(i)
        j = int(j)
        value += pow((r - np.matmul(U[:, i].T, P[:, j])), 2) + lr * (sum([pow(np.linalg.norm(U[:, i]), 2) for i in range(i)]) + sum([pow(np.linalg.norm(P[:, j]), 2) for j in range(j)]))
    return value


def create_matrices(group, d, top_n, n, data_path=None, titles=None):
    idxs = {}

    if titles is not None:
        for title in titles:
            for idx in title["Id"]:
                if idx in idxs:
                    idxs[idx] += 1
                else: 
                    idxs[idx] = 1

    elif data_path is not None:
        titles = []
        data = pd.read_csv(data_path, index_col=False)
        data = data[data["Group"] == group]
        top_products = data["Title"].value_counts()[:n].index.tolist()
        for i in range(n):
            titles.append(data.loc[data["Title"] == top_products[i]])
        
        for title in titles:
            for idx in title["Id"]:
                if idx in idxs:
                    idxs[idx] += 1
                else: 
                    idxs[idx] = 1

            
    else:
        return "data_path or titles required"

    sorted_idx = sorted(idxs.items(), key=operator.itemgetter(1))     
    top_idx = sorted_idx[-top_n:]
    
    R = np.zeros(shape=(top_n, len(titles)))
    for i, title in enumerate(titles):
        indexes = [id for id in title["Id"]]
        for j, idx in enumerate(top_idx):
            if idx[0] in indexes:
                R[j, i] = title["Ratings"].loc[title["Id"] == idx[0]][-1:]

    for i, title in enumerate(titles):
        indexes = indexes = [id for id in title["Id"]]


    U = np.zeros(shape=(d, R.shape[0]))
    P = np.zeros(shape=(d, R.shape[1]))

    return R, U, P


def recommendation(R, U, P, l, lr, d, p = 1):   
    #R, U, P, User_id, Product_id, df = create_matrix(data_path, n, d, category)
    
    R_up = create_R_up(R)
    U = np.random.uniform(0, 1, (U.shape[0], U.shape[1]))
    P = np.random.uniform(0, 1, (P.shape[0], P.shape[1]))
    
    U = np.matrix(U)
    P = np.matrix(P)
    R = np.matrix(R)
    for i in range(l):
        for k in range(U.shape[1]):
            I = np.flatnonzero(R[k])
            A = np.dot(P[:, I], P[:, I].T) + lr * np.eye(d)
            V = np.sum([R[k, j] * P[:, j] for j in I], axis=0)
            G = matrix(A, V)
            U[:, k] = G.Gauss()
            
        for k in range(P.shape[1]):
            I = np.flatnonzero(R[:, k])
            B = np.dot(U[:, I], U[:, I].T) + lr * np.eye(d)
            V = np.sum([R[j, k] * U[:, j] for j in I], axis=0)
            G = matrix(B, V)
            P[:, k] = G.Gauss()
            
        value = objective_function(R_up, U, P, lr)
        if p == 1:
            if l > 20:
                if i % 10 == 0:
                    print(value)
            else:
                print(value)
    return R, U, P

"""
def GEPP(A, b, doPricing = False):
    '''
    Gaussian elimination with partial pivoting.
    
    input: A is an n x n numpy matrix
           b is an n x 1 numpy array
    output: x is the solution of Ax=b 
            with the entries permuted in 
            accordance with the pivoting 
            done by the algorithm
    post-condition: A and b have been modified.
    '''
    n = len(A)
    if b.size != n:
        raise ValueError("Invalid argument: incompatible sizes between"+
                         "A & b.", b.size, n)
    # k represents the current pivot row. Since GE traverses the matrix in the 
    # upper right triangle, we also use k for indicating the k-th diagonal 
    # column index.
    
    # Elimination
    for k in range(n-1):
        if doPricing:
            # Pivot
            maxindex = abs(A[k:,k]).argmax() + k
            if A[maxindex, k] == 0:
                raise ValueError("Matrix is singular.")
            # Swap
            if maxindex != k:
                A[[k,maxindex]] = A[[maxindex, k]]
                b[[k,maxindex]] = b[[maxindex, k]]
        else:
            if A[k, k] == 0:
                raise ValueError("Pivot element is zero. Try setting doPricing to True.")
        #Eliminate
        for row in range(k+1, n):
            multiplier = A[row,k]/A[k,k]
            A[row, k:] = A[row, k:] - multiplier*A[k, k:]
            b[row] = b[row] - multiplier*b[k]
    # Back Substitution
    x = np.zeros(n)
    for k in range(n-1, -1, -1):
        x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
    return np.asmatrix(x).T
"""