
import pandas as pd
from src.utils import read_data




enc_data = read_data('data/test.true.a', 'ENCSENTS')
dec_data = read_data('data/test.true.b', 'DECSENTS')

# enc_data['ENCSLEN'] = enc_data.ENCSENTS.apply(lambda l: len(l.split()))
# dec_data['DECSLEN'] = dec_data.DECSENTS.apply(lambda l: len(l.split()))

dataset = pd.concat([enc_data,dec_data], axis=1)
dataset.dropna(axis=0, inplace=True)

dataset.to_csv('data/dataset.csv', index=False, encoding='utf-8', sep='\t')
