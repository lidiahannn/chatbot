import pandas as pd
from src.datamanager import DataManager


dataset = pd.read_csv('data/dataset.csv', encoding='utf-8', sep='\t')
dataset.dropna(axis=0, inplace=True)


dm = DataManager(batch_size=25)

df = dm.initialize(dataset)

train, test = dm.train_test_split(df, test_size=0.2, random_state=None)

gen = dm.batch_gen(train)
X, xl, y, yl = next(gen)




