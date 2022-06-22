import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.python.ops.math_ops import reduce_mean

from config.config import Tacotron2Config
from model.layers.Encoder import EncConvLayer
from model.layers.Decoder import LSAttention, Prenet, Postnet
from model.layers.MelSpec import MelSpec


import numpy as np


class Encoder(tf.keras.layers.Layer):
    def __init__(self, config): 
        super().__init__()
    
        self.encoder_conv = tf.keras.Sequential([EncConvLayer(
            config["conv_layer"]["filter"],
            config["conv_layer"]["kernel_size"], 
            config["conv_layer"]["dropout_rate"]) 
            for i in range(config["conv_layer"]["n"])] )

        self.bidir_lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units=config["bi_lstm"]["units"],
                    return_sequences=True
                    )
                )
        self.config = config

    def call(self, x):

        mask = x._keras_mask 
        y = self.encoder_conv(x)
        y = self.bidir_lstm(y, mask=mask)  #propagates mask

        return y

class Decoder(tf.keras.layers.Layer):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        dc = config["decoder"]
        self.prenet = Prenet(
                dc["prenet"]["units"],
                dc["prenet"]["n"],
                dc["prenet"]["dropout_rate"])
        self.lsattention_layer = LSAttention(
                dc["lsattention"]["rnn_dim"],
                dc["lsattention"]["embed_dim"],
                dc["lsattention"]["att_dim"],
                dc["lsattention"]["att_n_filters"],
                dc["lsattention"]["att_ker_size"])
        
        self.att_rnn = tf.keras.layers.LSTMCell(dc["lsattention"]["att_dim"])
        self.dec_rnn = tf.keras.layers.LSTMCell(dc["dec_rnn_units"])

        self.lin_proj_dense = tf.keras.layers.Dense(config["n_mel_channels"] * config["n_frames_per_step"])
        self.gate_dense = tf.keras.layers.Dense(1, use_bias=True)
        self.postnet = Postnet(
                dc["postnet"]["filters"],
                dc["postnet"]["n"],
                config["n_mel_channels"],
                dc["postnet"]["kernel_size"],
                dc["postnet"]["dropout_rate"],
                config["n_frames_per_step"])
        """
        self.decode = tf.function(func=self.decode,
            input_signature=[
                tf.TensorSpec(shape=[None, self.config["decoder"]["prenet"]["units"]], dtype=tf.float32),
                tf.TensorSpec(shape=[None, None, self.config["encoder"]["char_embedding_size"]], dtype=tf.float32),
                tf.TensorSpec(shape=[None, None, self.config["decoder"]["lsattention"]["att_dim"]], dtype=tf.float32),
            ],
        )
        """

    def prepare_decoder(self, enc_out):
        
        self.lsattention_layer.prepare_attention(enc_out)

        dc = self.config["decoder"]
        batch_size = tf.shape(enc_out)[0]

        self.att_hidden = tf.zeros([batch_size, dc["lsattention"]["rnn_dim"]])
        self.att_cell = self.att_rnn.get_initial_state(None, batch_size, dtype=tf.float32)
        
        self.dec_hidden = tf.zeros([batch_size, dc["dec_rnn_units"]])
        self.dec_cell = self.dec_rnn.get_initial_state(None, batch_size, dtype=tf.float32)
    
        self.W_enc_out = self.lsattention_layer.process_memory(enc_out)

    def decode(self, mel_in, enc_out, W_enc_out, enc_out_mask):
        att_rnn_in = tf.concat([mel_in, self.lsattention_layer.att_context], -1)
        self.att_hidden, self.att_cell = self.att_rnn(att_rnn_in, self.att_cell)

        self.att_context = self.lsattention_layer(
                self.att_hidden, enc_out, W_enc_out, enc_out_mask)

        dec_input = tf.concat([self.att_hidden, self.att_context], -1)

        self.dec_hidden, self.dec_cell = self.dec_rnn(dec_input, self.dec_cell)

        dec_hidden_att_context = tf.concat([self.dec_hidden, self.att_context], 1)

        dec_output = self.lin_proj_dense(dec_hidden_att_context)
        gate_output = self.gate_dense(dec_hidden_att_context)

        return dec_output, gate_output

    #@tf.function
    def call(self, enc_out, mel_gt, enc_out_mask):

        
        #first mel frame input
        first_mel_frame = tf.zeros([1, tf.shape(enc_out)[0], self.config["n_mel_channels"] * self.config["n_frames_per_step"]])

        #reshape mel_gt in order to group frames by reduction factor
        #it implies that given mel spec length is a multiple of n_frames_per_step
        #(batch_size x n_mel_channels x L) -> (batch_size x n_mel_channels * n_frames_per_step x L // n_frames_per_step ) 
        mel_gt = tf.reshape(mel_gt, [tf.shape(mel_gt)[0], tf.shape(mel_gt)[-1] // self.config["n_frames_per_step"], -1])
        mel_gt = tf.transpose(mel_gt, perm=[1,0,2])
        mel_gt = tf.concat([first_mel_frame, mel_gt], 0)
        mel_gt = mel_gt[:-1]



        mel_gt = self.prenet(mel_gt)

        mels_size = mel_gt.shape[0]
        mels_out, gates_out = tf.TensorArray(tf.float32, size=mels_size), tf.TensorArray(tf.float32, size=mels_size)

        
        self.prepare_decoder(enc_out)

        for i in tf.range(mels_size):

            mel_in = mel_gt[i]
            mel_out, gate_out = self.decode(mel_in, enc_out, self.W_enc_out, enc_out_mask)
            mel_out = tf.reshape(mel_out, [-1, self.config["n_frames_per_step"], self.config["n_mel_channels"]])
            mel_out = tf.reshape(mel_out, [-1, self.config["n_frames_per_step"], self.config["n_mel_channels"]])
            mels_out = mels_out.write(i, mel_out)
            gates_out = gates_out.write(i, gate_out)


            """ 
            mels_out.append(mel_out)
            gates_out.append(gate_out)
            """

        #return tf.concat(mels_out, 1), tf.concat(gates_out, 1)
        mels_out = mels_out.stack()
        gates_out = gates_out.stack()

        batch_size = tf.shape(mels_out)[1]
        mels_out = tf.reshape(mels_out, [batch_size, -1, self.config["n_mel_channels"], 1])
        gates_out = tf.reshape(gates_out, [batch_size, -1])



        return mels_out, gates_out



class Tacotron2(tf.keras.Model):
    def __init__(self, config: Tacotron2Config) -> None:
        super(Tacotron2, self).__init__()

        self.tokenizer = tf.keras.layers.TextVectorization(split=self.tv_func)

        self.config = config

        self.encoder = Encoder(self.config["encoder"])
        self.decoder = Decoder(self.config)

        melconf = config["mel_spec"]
    def set_vocabulary(self, dataset, n_batch=64):
        self.tokenizer.adapt(dataset.batch(n_batch))
        self.char_embedding = tf.keras.layers.Embedding(self.tokenizer.vocabulary_size(), 
                self.config["encoder"]["char_embedding_size"],
                mask_zero=True)

    def call(self, batch, training=False):

        phon, mels, mels_len = batch
        mels = tf.transpose(mels, perm=[0,2,1])

        x = self.tokenizer(phon)
        x = self.char_embedding(x)
        y = self.encoder(x)
        tf.print(tf.shape(y))

        crop = tf.shape(mels)[2] - tf.shape(mels)[2]%self.config["n_frames_per_step"]#max_len must be a multiple of n_frames_per_step
        mels_len = tf.clip_by_value(mels_len, 0, crop)
        mels, gates = self.decoder(y, mels[:,:,:crop], y._keras_mask)

        residual = self.decoder.postnet(mels)
        mels_post = mels + residual
        
        return (mels, mels_post, mels_len), gates

    @staticmethod
    def tv_func(x):
        x = tf.strings.unicode_split(x, 'UTF-8')
        return x

    @staticmethod
    def criterion(y_pred, y_true):
        mels, gates = y_pred
        mels_pre, mels_post = mels
        mels_true, gates_true = y_true
        gates_true = tf.reduce_mean(gates_true, axis=1)

        loss = tf.reduce_mean(tf.square(mels_pre - mels_true)) + \
                tf.reduce_mean(tf.square(mels_post - mels_true)) + \
                tf.nn.sigmoid_cross_entropy_with_logits(gates_true, gates)
        return loss

    @staticmethod
    def ljspeech_map_func(x):
        x = tf.strings.split(x, sep='|')[1]
        return x

if __name__ == "__main__":
    tac_conf = Tacotron2Config("config/configs/tacotron2.yaml")
    tac = Tacotron2(tac_conf)

    ljspeech_text = tf.data.TextLineDataset(tac_conf["train_data"]["transcript_path"])
    tac.set_vocabulary(ljspeech_text.map(lambda x : tf.strings.split(x, sep='|')[0]))
    map_F = generate_map_func(tac_conf)
    ljspeech_text = ljspeech_text.map(map_F)

    x, y = next(iter(ljspeech_text.padded_batch(16, padding_values=((None, None), (None, 1.)) )))
    mels, gates = tac(x)
    
