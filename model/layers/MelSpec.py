import tensorflow as tf
import tensorflow_io as tfio

class MelSpec(tf.keras.layers.Layer):
    def __init__(
            self,
            frame_length=1024,
            frame_step=256,
            fft_length=None,
            sampling_rate=22050,
            num_mel_channels=80,
            freq_min=125,
            freq_max=7600,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.sampling_rate = sampling_rate
        self.num_mel_channels = num_mel_channels
        self.freq_min = freq_min
        self.freq_max = freq_max

        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=self.num_mel_channels,
                num_spectrogram_bins=self.frame_length // 2 + 1,
                sample_rate=self.sampling_rate,
                lower_edge_hertz=self.freq_min,
                upper_edge_hertz=self.freq_max,
                )
    
    def call(self, audio):
        stft = tf.signal.stft(
                tf.squeeze(audio, -1),
                self.frame_length,
                self.frame_step,
                self.fft_length,
                pad_end=True,                    )

        magnitude = tf.abs(stft)

        mel = tf.matmul(tf.square(magnitude), self.mel_filterbank)
        log_mel_spec = tfio.audio.dbscale(mel, top_db=80)
        return log_mel_spec

