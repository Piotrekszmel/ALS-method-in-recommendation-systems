import pandas as pd


data_path = "dataset/amazon-meta.txt"


def parse_text(text):
    text = open(data_path, "r")
    ratings = []
    idxs = []
    product_types = []
    titles = []
    final_list = []
    i = 0
    
    group_switch = 0
    title_switch = 0 
    for row in text:
        row = row.replace("\n", "")
        if 'group: ' in row:
            product_type = "".join(row.split(':')[1]).strip()
            product_types.append(product_type)
            group_switch = 1

        if 'title' in row:
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
                
        
        i += 1
    
    for title, product_type, rating, idx in zip(titles, product_types, ratings, idxs):
        final_list.append([product_type, title, idx, rating])
    
    return final_list


def add_rows(df, data_path):
    data = parse_text(data_path)
    for row in data:
        df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
    print('DataFrame Updated!')
    return df

df = pd.DataFrame(columns=['Group', 'Title', 'Id', 'Ratings'])

df = add_rows(df, data_path)

print(df.head(100))

