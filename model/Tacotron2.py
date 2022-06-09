import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import concat

from dataset import tv_func
from config.config import Tacotron2Config
from model.tokenizer import Tokenizer
from model.layers.Encoder import EncConvLayer
from model.layers.Decoder import LSAttention, Prenet
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

        y = self.encoder_conv(x)
        y = self.bidir_lstm(y)

        return y

class Decoder(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
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
        
        self.att_rnn = tf.keras.layers.LSTM(dc["lsattention"]["att_dim"])
        self.dec_rnn = tf.keras.layers.LSTM(dc["dec_rnn_units"], True)

        self.lin_proj = tf.keras.layers.Dense(config["n_mel_channels"] * config["n_frames_per_step"])
        self.gate_dense = tf.keras.layers.Dense(1)

    def decode(self, mel_in, enc_out, W_enc_out):
        att_rnn_in = tf.concat([mel_in, self.att_context], -1)
        self.att_hidden = self.att_rnn(att_rnn_in)

        self.att_weights_cat = tf.concat([self.att_weights, self.att_weights_cat],
                1)
        self.att_context, self.att_weights = self.attention_layer(
                self.att_hidden, enc_out, W_enc_out, self.att_weights_cat)
        self.att_weights_cat.append(self.att_weights)







    def __call__(self, enc_out, mel_gt, mel_len):
        
        #first mel frame input
        first_mel_frame = tf.zeros([1, enc_out.shape[0], config["n_mel_channels"] * config["n_frames_per_step"]])

        #reshape mel_gt in order to group frames by reduction factor
        #it implies that given mel spec length is a multiple of n_frames_per_step
        #(batch_size x n_mel_channels x L) -> (batch_size x n_mel_channels * n_frames_per_step x L // n_frames_per_step ) 
        mel_gt = tf.reshape(mel_gt, [mel_gt.shape[0], mel_gt.shape[-1] // config["n_frames_per_step"], -1])
        mel_gt = tf.transpose(mel_gt, perm=[0,1])
        mel_gt = tf.concat([first_mel_frame, mel_gt], 0)

        mel_gt = self.prenet(mel_gt)

        mels_out, gates_out, alignments = [], [], []

        
        W_enc_out = self.lsattention_layer.process_memory(enc_out)

        for mel_in in mel_gt :

            mel_out, gate_out, att_weights = self.decode(mel_in, enc_out, W_enc_out)
        
            mels_out.append(mel_out)
            gate_out.append(gate_out)
            alignments.append(att_weights)

        return mels_out, gates_out, alignments




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
                config["encoder"]["char_embedding_size"])
        self.encoder = Encoder(self.config["encoder"])

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

        return y

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

    output = tac(batch)
    
    print("output shape : ", output.shape)



