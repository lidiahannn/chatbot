import pandas as pd
from src.datamanager import DataManager

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

##########
# Config
##########
batch_size = 2
vocab_size = 5000
embedding_size = 10
cell_num = 10


##########
# Data Manager
##########

dataset = pd.read_csv('data/dataset.csv', encoding='utf-8', sep='\t')
dataset.dropna(axis=0, inplace=True)


dm = DataManager(batch_size=batch_size, vocab_size=vocab_size)

df = dm.initialize(dataset)

train, test = dm.train_test_split(df, test_size=0.2, random_state=None)

gen = dm.batch_gen(train)
X, xl, y, yl, w = next(gen)


##########
# MODEL
##########

# Placeholders
inputs = tf.placeholder(tf.int32, shape=(None, None), name='X') # shape=(batch, seq_len)
source_sequence_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='xl') #shape=(batch, )
decoder_outputs = tf.placeholder(dtype=tf.int32, shape = (None,None), name='y') # This is label
decoder_lengths = tf.placeholder(dtype=tf.int32, shape=(None,), name='yl')
target_weight = tf.placeholder(dtype=tf.float32, shape=(None,None), name='w')


# Embedding
embedding_encoder = tf.get_variable("embedding_encoder", [vocab_size, embedding_size])
# here "200" is the imput embedding size.
# embedding_encoder is weight matrix.

encoder_emb_inp = tf.nn.embedding_lookup(embedding_encoder, inputs)

embedding_decoder = tf.get_variable("embedding_decoder", [vocab_size, embedding_size])
decoder_emb_inp = tf.nn.embedding_lookup(embedding_decoder, inputs)


# Encoder
# Build RNN cell
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_num) #here the "200" is the number of units, it refers to the output embedding size.#

# Run Dynamix RNN
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp, sequence_length=source_sequence_length,
                                                   dtype=tf.float32, time_major=False)


# Decoder

# Build RNN cell
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_num)

# Helper
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=False)

# Decoder
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=None)

# Dynamic decoding
outputs,state,_ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output

projection_layer = layers_core.Dense(vocab_size,use_bias=False)
#5000 is target lang vocab size, can be defined as a variable, same as source lang vocab in encoder

# Loss
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = decoder_outputs, logits = logits)
train_loss = (tf.reduce_sum(crossent * target_weight)/ batch_size)

# Compute and optimize Gradient
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)

# Optimization#

optimizer = tf.train.AdamOptimizer(0.0002)
update_step = optimizer.apply_gradients(zip(clipped_gradients,params))


def feed_dict(X, xl, y, yl, w):
    d = {inputs: X}
    d.update({source_sequence_length: xl})
    d.update({decoder_outputs: y})
    d.update({decoder_lengths: yl})
    d.update({target_weight: w})
    return d


init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)


gen = dm.batch_gen(train)
train_inputs = next(gen)

sess.run(outputs, feed_dict=feed_dict(*train_inputs))



