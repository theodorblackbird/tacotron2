from tensorflow.keras.layers import Layer, Dense, Conv1D
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.core import Dropout

import tensorflow as tf
class Prenet(Layer):
    def __init__(self, units, n, dropout_rate):
        super().__init__()
        self.layers = tf.keras.Sequential([Dense(units) for i in range(n)])
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def __call__(self, x):
        return self.dropout(self.layers(x))

class LSAttention(tf.keras.layers.Layer):

    def __init__(self, 
            rnn_dim,
            embed_dim,
            att_dim,
            att_n_filters,
            att_ker_size):
        super(LSAttention, self).__init__()

        self.query_dense = Dense(rnn_dim, att_dim, 
                use_bias=False)
        self.memory_dense = Dense(embed_dim, att_dim, 
                use_bias=False)

        self.location_dense = Dense(att_n_filters, att_dim,
                use_bias=False)
        self.location_conv = Conv1D(2, att_n_filters, 
                padding=int((att_ker_size - 1)/2),
                use_bias=False,
                stride=1,
                dilation=1)
        self.energy_dense(att_dim, 1, bias=False)

    def process_memory(self, memory):
        return self.memory_dense(memory)

    
    def alignement_score(self, query, W_memory, cum_att_weights):
        
        W_query = self.query_dense(tf.expand_dims(query, 1))
        W_att_weights = self.location_conv(cum_att_weights)
        W_att_weights = self.location_dense(tf.transpose(W_att_weights, perm=[1,2]))
        alignment = self.energy_dense(tf.math.tanh(W_query + W_att_weights + W_memory))
        return tf.squeeze(alignment)


    def __call__(self, att_hs, memory, W_memory, cum_att_weights):
        alignment = self.alignement_score(att_hs, W_memory, cum_att_weights)
        att_weights = tf.nn.softmax(alignment, axis=1)
        att_context = tf.matmul(tf.expand_dims(att_weights, 1), memory)
        att_context = tf.squeeze(att_context)
        return att_context, att_weights



