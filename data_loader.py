import pandas as pd
import numpy as np 


def parse_text(size, data_path):
    text = open(data_path, "r")
    ratings = []
    idxs = []
    product_types = []
    titles = []
    group_switch = 0
    title_switch = 0 
    
    for row in text:
        row = row.replace("\n", "")
        if len(set(titles)) > 0:
            if len(set(titles)) >= size and len(ratings) / len(set(titles)) == size:
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
    
    final_list = []
    products = []
    for title, product_type, rating, idx in zip(titles, product_types, ratings, idxs):
        final_list.append([product_type, title, idx, rating])
        if title not in products:
            products.append(title)
        if len(products) == size:
            break 
    
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
    
    return R, U, P, User_id, Product_id, df

