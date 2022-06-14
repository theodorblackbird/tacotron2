import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import concat

from config.config import Tacotron2Config
from model.tokenizer import Tokenizer
from model.layers.Encoder import EncConvLayer
from model.layers.Decoder import LSAttention, Prenet, Postnet
from model.layers.MelSpec import MelSpec

from preprocess.gruut_phonem import gruutPhonem

class Encoder(tf.keras.layers.Layer):
    def __init__(self, config):
        
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

    def __call__(self, x):

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
                dc["postnet"]["dropout_rate"])


    def prepare_decoder(self, enc_out):
        
        self.lsattention_layer.prepare_attention(enc_out)

        dc = self.config["decoder"]
        batch_size = enc_out.shape[0]

        self.att_hidden = tf.zeros([batch_size, dc["lsattention"]["rnn_dim"]])
        self.att_cell = self.att_rnn.get_initial_state(None, batch_size, dtype=tf.float32)
        
        self.dec_hidden = tf.zeros([batch_size, dc["dec_rnn_units"]])
        self.dec_cell = self.dec_rnn.get_initial_state(None, batch_size, dtype=tf.float32)
        
        self.W_enc_out = self.lsattention_layer.process_memory(enc_out)


    def decode(self, mel_in, enc_out, W_enc_out):
        att_rnn_in = tf.concat([mel_in, self.lsattention_layer.att_context], -1)
        self.att_hidden, self.att_cell = self.att_rnn(att_rnn_in, self.att_cell)

        self.att_context = self.lsattention_layer(
                self.att_hidden, enc_out, W_enc_out)

        dec_input = tf.concat([self.att_hidden, self.att_context], -1)

        self.dec_hidden, self.dec_cell = self.dec_rnn(dec_input, self.dec_cell)

        dec_hidden_att_context = tf.concat([self.dec_hidden, self.att_context], 1)

        dec_output = self.lin_proj_dense(dec_hidden_att_context)
        gate_output = self.gate_dense(dec_hidden_att_context)

        return dec_hidden_att_context, gate_output



    def __call__(self, enc_out, mel_gt):
        
        #first mel frame input
        first_mel_frame = tf.zeros([1, enc_out.shape[0], self.config["n_mel_channels"] * self.config["n_frames_per_step"]])

        #reshape mel_gt in order to group frames by reduction factor
        #it implies that given mel spec length is a multiple of n_frames_per_step
        #(batch_size x n_mel_channels x L) -> (batch_size x n_mel_channels * n_frames_per_step x L // n_frames_per_step ) 
        mel_gt = tf.reshape(mel_gt, [mel_gt.shape[0], mel_gt.shape[-1] // self.config["n_frames_per_step"], -1])
        mel_gt = tf.transpose(mel_gt, perm=[1,0,2])
        mel_gt = tf.concat([first_mel_frame, mel_gt], 0)

        mel_gt = self.prenet(mel_gt)

        mels_out, gates_out = [], []
        
        self.prepare_decoder(enc_out)
        for mel_in in mel_gt :

            mel_out, gate_out = self.decode(mel_in, enc_out, self.W_enc_out)
        
            mels_out.append(mel_out)
            gates_out.append(gate_out)
        return tf.stack(mels_out, 0), tf.stack(gates_out, 0)




class Tacotron2(tf.keras.Model):
    def __init__(self, config: Tacotron2Config) -> None:
        super(Tacotron2, self).__init__()
         
        text = tf.data.TextLineDataset(config["train_data"]["transcript_path"])
        text = text.map(ljspeech_map_func)

        self.tokenizer = tf.keras.layers.TextVectorization(split=self.tv_func)
        self.tokenizer.adapt(text.batch(64))
        
        self.phonem = gruutPhonem()
        self.config = config
        self.char_embedding = tf.keras.layers.Embedding(self.tokenizer.vocabulary_size(), 
                config["encoder"]["char_embedding_size"],
                mask_zero=True)
        self.encoder = Encoder(self.config["encoder"])
        self.decoder = Decoder(self.config)

        melconf = config["mel_spec"]
        self.melspec = MelSpec(
                melconf["frame_length"],
                melconf["frame_step"],
                melconf["fft_length"],
                melconf["sampling_rate"],
                melconf["n_mel_channels"],
                melconf["freq_min"],
                melconf["freq_max"])

    def __call__(self, x):

        x = self.tokenizer(x)
        x = self.char_embedding(x)
        y = self.encoder(x)

        mel_test = tf.ones([x.shape[0], 80, 1000])
        mels, gates = self.decoder(y, mel_test)
        print("mel shape : ", mels.shape)

        residual = self.decoder.postnet(mels)

        mels = mels + residual
        return mels, gates

    @staticmethod
    def tv_func(x):
        x = tf.strings.unicode_split(x, 'UTF-8')
        return x



def ljspeech_map_func(x):
    x = tf.strings.split(x, sep='|')[1]
    return x

if __name__ == "__main__":
    tac_conf = Tacotron2Config("config/configs/tacotron2.yaml")
    tac = Tacotron2(tac_conf)

    
    ljspeech_text = tf.data.TextLineDataset(tac_conf["train_data"]["transcript_path"])
    ljspeech_text = ljspeech_text.map(ljspeech_map_func)

    batch = next(iter(ljspeech_text.batch(16)))
    print(bytes.decode(batch.numpy()[0]))
    mels, gates = tac(batch)
    
    print("output shape : ", mels[0].shape)



