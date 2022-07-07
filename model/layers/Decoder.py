from tensorflow.keras import Sequential
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.core import Dropout
import torch
import torch.nn.functional as F
from torch import nn

class Prenet(nn.Module):
    def __init__(self, in_feat, units, n, dropout_rate):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(in_feat, units))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        for i in range(n-1):
            self.layers.append(nn.Linear(units, units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class LSAttention(nn.Module):

    def __init__(self,
            rnn_dim,
            embed_dim,
            att_dim,
            att_n_filters,
            att_ker_size):
        super(LSAttention, self).__init__()

        self.query_dense = nn.Linear(
                rnn_dim,
                att_dim, 
                use_bias=False)
        self.memory_dense = nn.Linear(
                embed_dim,
                att_dim, 
                use_bias=False)

        self.location_dense = nn.Linear(
                att_n_filters,
                att_dim,
                use_bias=False)
        self.location_conv = nn.Conv1D(
                2,
                att_n_filters, 
                att_ker_size,
                padding=int((att_ker_size-1)/2),
                use_bias=False)

        self.energy_dense = nn.Linear(
                att_dim,
                1, 
                use_bias=False)


    def prepare_attention(self, batch):
        batch_size = batch.shape[0]
        max_len = batch.shape[1]
        encoder_dim = batch.shape[2]

        self.att_weights = torch.zeros(batch_size, max_len)
        self.cum_att_weights = torch.zeros_like(self.att_weights)
        self.att_context = torch.zeros(batch_size, encoder_dim)
    
    def process_memory(self, memory):
        return self.memory_dense(memory)

    def alignment_score(self, query, W_memory):
        
        cat_att_weights = torch.concat([self.att_weights.unsqueeze(1), self.cum_att_weights.unsqueeze(1)],
                1)
        cat_att_weights = torch.permute(cat_att_weights, (0,2,1))
        
        W_query = self.query_dense(query.unsqueeze(1))
        W_att_weights = self.location_conv(cat_att_weights)
        W_att_weights = self.location_dense(W_att_weights)
        alignment = self.energy_dense(torch.tanh(W_query + W_att_weights + W_memory))
        return alignment.squeeze(-1)

    def forward(self, att_hs, memory, W_memory, memory_mask):


        alignment = self.alignment_score(att_hs, W_memory)
        alignment.data.masked_fill_(memory_mask,-float("inf"))
        att_weights = F.softmax(alignment, axis=1)
        att_context = torch.bmm(att_weights.unsqueeze(1), memory)
        att_context = att_context.squeeze(1)


        self.cum_att_weights += att_weights
        return att_context, alignment

class DecConvLayer(nn.Module):
    def __init__(self,
            n_mel_channels
            filters,
            kernel_size,
            dropout_rate) -> None:
        super().__init__()
        self.conv = nn.Conv1D(
                n_mel_channels
                filters,
                kernel_size,
                padding=int((kernel_size - 1) / 2))
        self.bn = nn.BatchNorm1d(filters)
        self.dropout = nn.Dropout(
                p=dropout_rate)
        self.support_masking = True
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = tf.nn.relu(y)
        y = self.dropout(y)
        return y


class Postnet(nn.Module):
    def __init__(self,
            filters, 
            n,
            n_mel_channels,
            kernel_size,
            dropout_rate,
            n_frames_per_step):
        super().__init__()
        self.layers = []
        for i in range(0, n):
            self.layers.append(DecConvLayer(
                n_mel_channels,
                filters, 
                kernel_size, 
                dropout_rate))
        self.layers.append(nn.Linear(
            filters,
            n_mel_channels))
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)




