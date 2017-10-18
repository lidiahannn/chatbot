import numpy as np
from collections import Counter
from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


class DataManager(object):

    def __init__(self, batch_size, vocab_size=None, sos_symbol=1, eos_symbol=2, uwk_symbol=3, start_idx=4, enc_col='ENCSENTS', dec_col='DECSENTS'):
        self.batch_size = batch_size
        self.sos_symbol = sos_symbol
        self.eos_symbol = eos_symbol
        self.uwk_symbol = uwk_symbol
        self.start_idx = start_idx


        self.enc_col = enc_col
        self.dec_col = dec_col
        self._vocab_size = vocab_size

        self.ready = None


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

    def initialize(self, df, interval_size=25, on='ENCSENTS_TL'):
        _df = df.copy()
        self._freq_dist(df)
        self.vocab_size = len(self.counter) if self._vocab_size is None else self._vocab_size
        self._w_idx()
        _df[self.enc_col+'_TOK'] = _df[self.enc_col].str.split()
        _df[self.dec_col+'_TOK'] = _df[self.dec_col].str.split()
        _df[self.enc_col+'_TL'] = _df[self.enc_col+'_TOK'].apply(lambda l: len(l))
        _df[self.dec_col+'_TL'] = _df[self.dec_col+'_TOK'].apply(lambda l: len(l))
        return self.bucketing(_df, interval_size, on)

    def bucketing(self, df, interval_size, on):
        """
        Give a bucket id to each example based on interval_size and
        :param on: column name of token length in dataframe
        :return: dataframe same as input df with a new column of bucket ids
        """
        _df = df.copy()
        max_len = df[on].max()
        self.intervals = {i:(interval_size*(i-1),interval_size*i) for i in range(1, int(np.ceil(max_len/interval_size))+1)}
        _df['BUCKET'] = _df[on].apply(lambda x: [i for i,intv in self.intervals.items() if intv[0]<x<=intv[1]][0])
        return _df


    def tok2sym(self, tokens):
        """
        Input a pd.Series of lists as tokenized sentences, return symbolized Series
        :param tokens: pd.Series, a column of the tokenized dataframe
        :return:
        """
        symbols = tokens.apply(lambda l: [self.sos_symbol] +
                                         [self.w_idx[w] if w in self.w_idx else self.uwk_symbol for w in l ] +
                                         [self.eos_symbol])
        return symbols

    def pad_by_buckets(self, init_df):
        buckets = sorted(self.intervals.keys())
        bucket = {}
        for b in buckets:
            _df = init_df.loc[init_df.BUCKET==b][[self.enc_col+'_TOK',self.dec_col+'_TOK',self.dec_col+'_TL']]
            if not _df.empty:
                _enc_symbs = self.tok2sym(_df[self.enc_col+'_TOK'])
                _dec_symbs = self.tok2sym(_df[self.dec_col+'_TOK'])
                enc_mlen = max(self.intervals[b])
                dec_mlen = _df[self.dec_col+'_TL'].max()
                enc_symbs = pad_sequences(_enc_symbs, maxlen=enc_mlen)
                dec_symbs = pad_sequences(_dec_symbs, maxlen=dec_mlen)
                bucket.update({b: (enc_symbs,dec_symbs)})
        return bucket

    def train_test_split(self, df, test_size, random_state=None):
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
        return train, test

    def batch_gen(self, df):
        if self.ready is None:
            self.ready = self.pad_by_buckets(df)

        for b,(X,y) in self.ready.items():
            X_size, X_len = X.shape
            y_size, y_len = y.shape
            xl = np.repeat([X_len, ], self.batch_size).astype('int32')
            yl = np.repeat([y_len, ], self.batch_size).astype('int32')
            assert X_size==y_size, 'encoder/decoder input size must be same, different in bucket %s.' % b
            size = X_size
            n_batchs = size // self.batch_size if size % self.batch_size == 0 else size // self.batch_size + 1
            current = 0
            for i in range(n_batchs):
                next_batch = self.batch_size * (i + 1)
                if i == n_batchs - 1:
                    _X = X[current:]
                    _y = y[current:]
                else:
                    _X = X[current:next_batch]
                    _y = y[current:next_batch]
                yield _X, xl, _y, yl, (_y!=0).astype('float32')
                current = next_batch



