from librosa import stft
import librosa
import numpy as np

class MelSpec:
    def __init__(
            self,
            frame_length=1024,
            frame_step=256,
            fft_length=None,
            sampling_rate=22050,
            num_mel_channels=80,
            freq_min=125,
            freq_max=7600,
            ref_level_db=20.,
            min_level_db=-100.,
            max_norm=4.,
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
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.max_norm = max_norm

        self.mel_filterbank = librosa.filters.mel(
                self.sampling_rate,
                self.frame_length,
                n_mels=self.num_mel_channels,
                fmin=self.freq_min,
                fmax=self.freq_max)
    
    def __call__(self, audio):
        S = stft(
                audio,
                n_fft=self.frame_length,
                hop_length=self.frame_step,
                window="hann",
                center=True,
                )
        S = np.dot(self.mel_filterbank, np.abs(S))
        S = 20. * np.log10(np.maximum(1e-5, S))
        S = S - self.ref_level_db
        S = (S - self.min_level_db)/-self.min_level_db
        S = np.clip(S, 0., 4.)
        return S

