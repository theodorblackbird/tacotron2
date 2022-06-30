from tensorflow.data import Dataset
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from config.config import Tacotron2Config
from model.layers.MelSpec import MelSpec
import librosa

class ljspeechDataset(object):
    def __init__(self, conf, train_conf) -> None:
        super().__init__()
        self.conf = conf
        self.train_conf = train_conf
        msc = conf["mel_spec"]
        self.mel_spec_gen = MelSpec(
                msc["frame_length"],
                msc["frame_step"],
                None,
                msc["sampling_rate"],
                msc["n_mel_channels"],
                msc["freq_min"],
                msc["freq_max"])
        stats = np.loadtxt(train_conf["data"]["statistics"])
        self.mean = tf.constant(stats[0:msc["n_mel_channels"]], dtype=tf.float32)
        self.std = tf.math.sqrt(tf.constant(stats[msc["n_mel_channels"]:], dtype=tf.float32))

    

    def __call__(self, x):
        
        split = tf.strings.split(x, sep='|')
        name = split[0]
        phon = split[1]
        path = self.train_conf["data"]["audio_dir"] + "/"+ name + ".wav"
        raw_audio = tf.io.read_file(path)
        audio, sr = tf.audio.decode_wav(raw_audio)
        audio, _ = tf.py_function(librosa.effects.trim, [audio], [tf.float32, tf.int32])


        mel_spec = self.mel_spec_gen(audio)


        #mel_spec = (mel_spec - self.mean)/self.std
        crop = tf.shape(mel_spec)[0] - tf.shape(mel_spec)[0]%self.conf["n_frames_per_step"]#max_len must be a multiple of n_frames_per_step
        mel_len = len(mel_spec)
        gate = tf.zeros(mel_len-1)
        gate = tf.concat( (gate, [1.]), axis=0)

        return (phon, mel_spec, mel_len), (mel_spec, gate)

if __name__ == "__main__":

    from model.Tacotron2 import Tacotron2
    import matplotlib.pyplot as plt
    import numpy as np
    conf = Tacotron2Config("config/configs/tacotron2_laptop.yaml")

    tac = Tacotron2(conf)
    ljspeech_text = tf.data.TextLineDataset(conf["train_data"]["transcript_path"])
    dataset_mapper = ljspeechDataset(conf)
    ljspeech = ljspeech_text.map(dataset_mapper).shuffle(100)
    x, y = next(iter(ljspeech))
    phon, mel, gate = x
    print(mel.shape)
    mel = tf.transpose(mel)
    fig, ax = plt.subplots(1)
    fig.set_figwidth(20)
    fig.set_figheight(6)
    ax.imshow(mel)
    plt.show()
    print(np.max(mel))


