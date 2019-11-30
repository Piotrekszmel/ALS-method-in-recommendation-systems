import pandas as pd
import numpy as np 

data_path = "dataset/amazon-meta.txt"


def parse_text(text, size):
    text = open(data_path, "r")
    ratings = []
    idxs = []
    product_types = []
    titles = []
    final_list = []
    group_switch = 0
    title_switch = 0 
    for row in text:
        row = row.replace("\n", "")
        
        if len(ratings) >= size:
            break
        
        
        if 'group: ' in row:
            product_type = "".join(row.split(':')[1]).strip()
            product_types.append(product_type)
            group_switch = 1

        if 'title: ' in row:
            row = row.split()
            title = " ".join(word for word in row[1:])
            titles.append(title)
            title_switch = 1  
        
        if 'rating:' in row and 'avg rating' not in row:
            row = row.split(":")
            rating = row[2][1].strip()
            customer_id = row[1][:-6].strip()
            idxs.append(customer_id)
            ratings.append(rating)
            if group_switch == 0:
                product_types.append(product_type)
            else: 
                group_switch = 0
                
            if title_switch == 0:
                titles.append(title)
            else: 
                title_switch = 0
        
    
    for title, product_type, rating, idx in zip(titles, product_types, ratings, idxs):
        final_list.append([product_type, title, idx, rating])
    
    return final_list


def add_rows(df, data_path, size):
    data = parse_text(data_path, size)
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
    
    return R, U, P, User_id, Product_id, df

R, U, P, User_id, Product_id, df = create_matrix(data_path, 100, 3)

print(R.shape)
print(U.shape)
print(P.shape)







