import pandas as pd
import numpy as np 


def parse_text(size, data_path):
    text = open(data_path, "r")
    titles = []
    final_list = []
    rating = 0
    
    for row in text:
        row = row.replace("\n", "")
        
        if len(set(titles)) == size:
            break
        
        if 'group: ' in row:
            product_type = "".join(row.split(':')[1]).strip()
            
        if 'title: ' in row:
            row = row.split()
            title = " ".join(word for word in row[1:])
            
        if 'rating:' in row and 'avg rating' not in row:
            row = row.split(":")
            rating = row[2][1].strip()
            customer_id = row[1][:-6].strip()
        
        if rating != 0:
            final_list.append([product_type, title, customer_id, rating])
            titles.append(title)
            
            rating = 0
         
    return final_list


def add_rows(df, data_path, size):
    data = parse_text(size, data_path)
    for row in data:
        df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
    print('DataFrame Updated!')
    
    return df


def create_matrix(data_path, size, d):
    df = pd.DataFrame(columns=['Group', 'Title', 'Id', 'Ratings'])
    df = add_rows(df, data_path, size)
    users = set(df["Id"])
    df["index"] = range(0, len(df))
    
    User_id = {key: val for val, key in enumerate(users)}
    Product_id = {key : val for val, key in enumerate(set(df["Title"]))}

    R = np.zeros(shape=(len(users), len(Product_id)))
    U = np.zeros(shape=(d, len(users)))
    P = np.zeros(shape=(d, len(Product_id)))

    for _, row in df.iterrows():
        R[User_id[row["Id"]]][Product_id[row["Title"]]] = row["Ratings"]
    
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
