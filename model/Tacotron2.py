from ctypes import alignment

from config.config import Tacotron2Config
from model.layers.Encoder import EncConvLayer
from model.layers.Decoder import LSAttention, Prenet, Postnet
from model.layers.MelSpec import MelSpec

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Encoder(torch.nn.Module):
    def __init__(self, config): 
        super().__init__()

        self.encoder_conv = torch.nn.Sequential(*[EncConvLayer(
            config["char_embedding_size"],
            config["char_embedding_size"],
            config["conv_layer"]["kernel_size"], 
            config["conv_layer"]["dropout_rate"]) 
            for i in range(config["conv_layer"]["n"])] )


        self.bidir_lstm = torch.nn.LSTM(
                config["char_embedding_size"],
                config["bi_lstm"]["units"],
                bidirectional=True)
        self.config = config

    def forward(self, embed, embed_lens):
        y = self.encoder_conv(embed.transpose(1,2)).transpose(1,2)
        y = torch.nn.utils.rnn.pack_padded_sequence(
                embed,
                embed_lens,
                batch_first=True,)
        y, (h,c) = self.bidir_lstm(y)
        y, _ = torch.nn.utils.rnn.pad_packed_sequence(
                y,
                batch_first=True)
        return y

class Decoder(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        dc = config["decoder"]
        ec = config["encoder"]
        self.prenet = Prenet(
                config["n_mel_channels"],
                dc["prenet"]["units"],
                dc["prenet"]["n"],
                dc["prenet"]["dropout_rate"])
        self.lsattention_layer = LSAttention(
                dc["lsattention"]["rnn_dim"],
                ec["bi_lstm"]["units"]*2,
                dc["lsattention"]["att_dim"],
                dc["lsattention"]["att_n_filters"],
                dc["lsattention"]["att_ker_size"])

        self.att_rnn = nn.LSTMCell(
                dc["prenet"]["units"] + ec["bi_lstm"]["units"]*2,
                dc["lsattention"]["rnn_dim"])
        self.dec_rnn = nn.LSTMCell(dc["lsattention"]["att_dim"] + ec["bi_lstm"]["units"]*2,
                dc["dec_rnn_units"])

        self.lin_proj_dense = nn.Linear(dc["dec_rnn_units"],
                config["n_mel_channels"] * config["n_frames_per_step"])
        self.gate_dense = nn.Linear(dc["dec_rnn_units"] + ec["bi_lstm"]["units"]*2,
                1, bias=True)
        self.postnet = Postnet(
                dc["postnet"]["filters"],
                dc["postnet"]["n"],
                config["n_mel_channels"],
                dc["postnet"]["kernel_size"],
                dc["postnet"]["dropout_rate"],
                config["n_frames_per_step"])

    def prepare_decoder(self, enc_out):

        self.lsattention_layer.prepare_attention(enc_out)

        dc = self.config["decoder"]
        batch_size = enc_out.shape[0]

        self.att_hidden = torch.zeros([batch_size, dc["lsattention"]["rnn_dim"]])
        self.att_cell = torch.zeros([batch_size, dc["lsattention"]["rnn_dim"]])

        self.dec_hidden = torch.zeros([batch_size, dc["dec_rnn_units"]])
        self.dec_cell = torch.zeros([batch_size, dc["dec_rnn_units"]])


        self.W_enc_out = self.lsattention_layer.process_memory(enc_out)

    def decode(self, mel_in, enc_out, W_enc_out, enc_out_mask):
        att_rnn_in = torch.concat((mel_in, self.lsattention_layer.att_context), -1)
        self.att_hidden, self.att_cell = self.att_rnn(att_rnn_in, (self.att_hidden, self.att_cell))

        self.att_hidden = F.dropout(self.att_hidden, 
        self.config["decoder"]["lsattention"]["rnn_dropout_rate"])

        self.att_context, alignment = self.lsattention_layer(
                self.att_hidden, enc_out, W_enc_out, enc_out_mask)

        dec_input = torch.concat((self.att_hidden, self.att_context), -1)
        print(dec_input.shape)
        print(self.att_hidden.shape, self.att_context.shape)

        self.dec_hidden, self.dec_cell = self.dec_rnn(dec_input,(self.dec_hidden, self.dec_cell))
        
        self.dec_hidden = F.dropout(self.dec_hidden,
                self.config["decoder"]["dec_rnn_dropout_rate"])

        dec_hidden_att_context = torch.concat((self.dec_hidden, self.att_context), 1)

        dec_output = self.lin_proj_dense(dec_hidden_att_context)
        gate_output = self.gate_dense(dec_hidden_att_context)

        return dec_output, gate_output, alignment

    def forward(self, enc_out, mel_gt, tokens_len):

        batch_size = mel_gt.shape[0]
        mel_gt = mel_gt.transpose(1,2)

        #first mel frame input
        first_mel_frame = torch.zeros((1, enc_out.shape[0], self.config["n_mel_channels"] * self.config["n_frames_per_step"]))

        #reshape mel_gt in order to group frames by reduction factor
        #it implies that given mel spec length is a multiple of n_frames_per_step
        #(batch_size x n_mel_channels x L) -> (batch_size x n_mel_channels * n_frames_per_step x L // n_frames_per_step)
        mel_gt = torch.reshape(mel_gt, (batch_size, mel_gt.shape[-1] // self.config["n_frames_per_step"], -1))
        mel_gt = mel_gt.transpose(1,0)
        mel_gt = torch.concat((first_mel_frame, mel_gt[:-1] ), 0)

        mel_gt = self.prenet(mel_gt)

        mels_size = mel_gt.shape[0]
        mels_out, gates_out, alignments = [], [], []



        self.prepare_decoder(enc_out)

        max_len = torch.max(tokens_len)
        enc_out_mask = torch.arange(max_len)[None, :] < tokens_len[:, None]

        for i in range(mels_size):

            mel_in = mel_gt[i]
            mel_out, gate_out, alignment = self.decode(mel_in, enc_out, self.W_enc_out, enc_out_mask)

            mels_out.append(mel_out)
            gates_out.append(gate_out)
            alignments.append(alignment)

        mels_out = torch.stack(mels_out)
        gates_out = torch.stack(gates_out)
        alignments = torch.stack(alignments)

        mels_out = mels_out.transpose(1,0)
        alignments = alignments.transpose(1,0)
        mels_out = mels_out.reshape( (batch_size, -1, self.config["n_mel_channels"], 1))
        gates_out = gates_out.reshape((batch_size, -1))

        return mels_out, gates_out, alignments


class Tacotron2(torch.nn.Module):
    def __init__(self, config: Tacotron2Config, train_conf) -> None:
        super(Tacotron2, self).__init__()

        self.config = config
        self.train_config = config

        self.encoder = Encoder(self.config["encoder"])
        self.decoder = Decoder(self.config)

    def set_vocabulary(self, voc_size):
        self.embedding = torch.nn.Embedding(voc_size, self.config["encoder"]["char_embedding_size"])

    def forward(self, batch):

        tokens, mels, gates, mels_len, tokens_len = batch
        mels = mels.transpose(1,2)
        y = self.embedding(tokens)
        y = self.encoder(y, tokens_len)

        crop = mels.shape[2] - mels.shape[2]%self.config["n_frames_per_step"]#max_len must be a multiple of n_frames_per_step


        mels, gates, alignments = self.decoder(y, mels[:,:,:crop], tokens_len)

        residual = self.decoder.postnet(mels.squeeze(-1))
        mels_post = mels + residual.unsqueeze(-1)

        return (mels, mels_post, mels_len), gates, alignments


    def train_step(self, data):
        tokens, mels, gates, mel_lens, tokens_len = data
        mels, gates, alignments = self(data)
        mels, mels_post, mels_len = mels
        print("OK!")


        crop = true_mels.shape[1] - true_mels.shape[1]%self.config["n_frames_per_step"]#max_len must be a multiple of n_frames_per_step

        """
        compute loss
        """

        return {'loss': loss, 'pre_loss' : pre_loss, 'post_loss' : post_loss, 'gate_loss' : gate_loss}
