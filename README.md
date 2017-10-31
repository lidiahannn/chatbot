# Chatbot based on seq2seq model 
+ Tensorflow 1.3.0
+ Python 3.5

## Background
This toy chatbot is built with a 2 RNNs networks of one layer which uses LSTM as recurrent units. 
It comprises an encoder for input questions and a decoder for the output answers. 
The encoder receives the question as input, and then a boundary marker which indicates the 
transition from the encoding to the decoding. The decoder initializes with the last hidden state of 
the encoder and then predict the corresponding output answer.

## Dataset
`data/dataset.csv` A sample dataset of movie conversations.

`src.datamanager.DataManager` is used to preprocess data, including: 
+ symbolize tokens: convert tokens (words) into symbols (integers from `0` to `vocab_size`), where `0` stands
for padding, `1` stands for `sos` (start of sentence), `2` stands for `eos` (end of sentence), `3` stands for
`uwk` (unknown words); `vocab_size` is the total number of most frequent unique words in the corpus, words
outside of this range is considered as `uwk`.
+ bucketing and padding sentences: put sentence with similar length into same bucket and pad to same length
to avoid unnecessary computation.
+ iterate dataset for mini-batch training.

## Word embedding
Word embedding is needed to retrieve each of the words in the question and answer data sets. Word vectors are retrieved
from embedding matrix with shape `[vocab_size+start_idx, embedding_size]` according to symbols. 


## Model
`test/test.py` Script for building model graph and feed data for training.