import pandas as pd
from collections import Counter


dataset = pd.read_csv('data/dataset.csv', encoding='utf-8', sep='\t')
dataset.dropna(axis=0, inplace=True)


# 1. Count word frequency, create word 2 symbol mapping
# encoder, decoder should share same symbol and embedding matrix






class DataManager(object):

    def __init__(self,  vocab_size=None, start_idx=4, enc_col='ENCSENTS', dec_col='DECSENTS'):
        self.enc_col = enc_col
        self.dec_col = dec_col
        self.start_idx = start_idx
        self._vocab_size = vocab_size


    def _freq_dist(self, df):
        """
        Return Counter object of global vocabulary size, both encoder and
        decoder sentences.
        """
        # apply a split() function to all strings in the Series
        enc_sents = df[self.enc_col].str.split()
        dec_sents = df[self.dec_col].str.split()
        count_sents = enc_sents.append(dec_sents, ignore_index=True)
        counter = Counter()
        for idx,lst in count_sents.items():
            counter.update(lst)
        self.counter =  counter

    def _w_idx(self):
        """
        Create a dictionary with token as key, symbol as value
        """
        if self.vocab_size is None:
            w_idx = {w:i+self.start_idx for i,(w,c) in enumerate(self.counter.most_common())}
        else:
            w_idx = {w:i+self.start_idx for i,(w,c) in enumerate(self.counter.most_common(self.vocab_size))}
        self.w_idx = w_idx

    def initialize(self, df):
        self._freq_dist(df)
        self.vocab_size = len(self.counter) if self._vocab_size is None else self._vocab_size
        self._w_idx()



dm = DataManager(vocab_size=15)

dm.initialize(dataset)












