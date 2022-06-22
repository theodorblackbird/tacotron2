from tensorflow.data import Dataset
import tensorflow as tf
from config.config import Tacotron2Config
from model.layers.MelSpec import MelSpec
from os.path import join


tac_conf = Tacotron2Config("config/configs/tacotron2.yaml")

class ljspeechDataset(object):
    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf
        msc = conf["mel_spec"]
        self.mel_spec_gen = MelSpec(
                msc["frame_length"],
                msc["frame_step"],
                None,
                msc["sampling_rate"],
                msc["n_mel_channels"],
                msc["freq_min"],
                msc["freq_max"])

    def __call__(self, x):
        
        split = tf.strings.split(x, sep='|')
        name = split[0]
        phon = split[1]
        path = self.conf["train_data"]["audio_dir"] + "/"+ name + ".wav"
        raw_audio = tf.io.read_file(path)
        audio, sr = tf.audio.decode_wav(raw_audio)
        mel_spec = self.mel_spec_gen(audio)
        gate = tf.zeros_like(mel_spec)
        mel_len = len(mel_spec)
        return (phon, mel_spec, mel_len), (mel_spec, gate)

"""
F = generate_map_func(tac_conf)
ljspeech_text = tf.data.TextLineDataset(tac_conf["train_data"]["transcript_path"])
ljspeech_text = ljspeech_text.map(F)
x, y = next(iter(ljspeech_text))

print(x, y.shape)
"""
