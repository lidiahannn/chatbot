import pandas as pd


def read_data(file, colname):
    with open(file, 'r', encoding='utf-8') as f:
        data = [[line.strip()] for line in f]
    df = pd.DataFrame(data=data, columns=[colname])
    return df



