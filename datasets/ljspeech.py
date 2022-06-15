from tensorflow.data import Dataset
import tensorflow as tf
from config.config import Tacotron2Config
from model.layers.MelSpec import MelSpec
from os.path import join


tac_conf = Tacotron2Config("config/configs/tacotron2.yaml")

def generate_map_func(conf):
    def map_func(x):
        
        split = tf.strings.split(x, sep='|')
        name = split[0]
        phon = split[1]
        path = conf["train_data"]["audio_dir"] + "/"+ name + ".wav"
        raw_audio = tf.io.read_file(path)
        audio, sr = tf.audio.decode_wav(raw_audio)
        msc = tac_conf["mel_spec"]
        mel_spec_gen = MelSpec(
                msc["frame_length"],
                msc["frame_step"],
                None,
                msc["sampling_rate"],
                msc["n_mel_channels"],
                msc["freq_min"],
                msc["freq_max"])
        mel_spec = mel_spec_gen(audio)
        mel_spec = tf.transpose(mel_spec, perm=[1,0])
        return phon, mel_spec
    return map_func

"""
F = generate_map_func(tac_conf)
ljspeech_text = tf.data.TextLineDataset(tac_conf["train_data"]["transcript_path"])
ljspeech_text = ljspeech_text.map(F)
x, y = next(iter(ljspeech_text))

print(x, y.shape)
"""
