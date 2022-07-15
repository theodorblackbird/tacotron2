import numpy as np
from config.config import Tacotron2Config
import librosa
from model.layers.MelSpec import MelSpec
from model.tokenizer import Tokenizer
import torch
import os
class ljspeechDataset(torch.utils.data.Dataset):

    def __init__(self, conf, train_conf) -> None:
        self.transcript_path = train_conf["data"]["transcript_path"]
        self.audio_path = train_conf["data"]["audio_dir"]
        self.conf = conf
        self.n_frames_per_step = conf["n_frames_per_step"]
        with open(self.transcript_path) as f:
            self.text = f.readlines()
        self.tokenizer = Tokenizer()
        ms = conf["mel_spec"]
        self.melspec = MelSpec(
            ms["frame_length"],
            ms["frame_step"],
            ms["fft_length"],
            ms["sampling_rate"],
            ms["n_mel_channels"],
            ms["freq_min"],
            ms["freq_max"],)
 
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        line = self.text[idx]
        line = line.split("|")
        locution_id, locution = line
        wav_path = os.path.join(self.audio_path, locution_id+".wav")
        y, sr = librosa.load(wav_path)
        mel = self.melspec(y)
        locution = locution.strip()
        tokens = self.tokenizer.encode(locution)
        
        crop = mel.shape[1] - mel.shape[1]%self.n_frames_per_step
        mel = mel[:,:crop]

        return tokens, mel


    def collate_fn(self, batch):
        
        tokens_lens, ids_sorted = torch.sort(
                torch.IntTensor(list(map(lambda x: len(x[0]), batch))),
                dim=0,
                descending=True)

        max_tokens_len = max(tokens_lens)
        mel_lens = list(map(lambda x: x[1].shape[1], batch))
        max_mel_len = max(mel_lens)
        mels, tokens, gates = [], [], []
        for i in ids_sorted :
            t, m = batch[i]
            padded_mel = torch.zeros(self.conf["mel_spec"]["n_mel_channels"], max_mel_len)
            padded_tokens = torch.zeros(max_tokens_len).int()
            padded_gate = torch.ones(max_mel_len)

            padded_mel[:, :m.shape[1]] = torch.from_numpy(m)
            padded_tokens[:len(t)] = torch.Tensor(t)
            padded_gate[:m.shape[1]-1] = 0

            mels.append(padded_mel)
            tokens.append(padded_tokens)
            gates.append(padded_gate)

        return torch.stack(tokens), torch.stack(mels), torch.stack(gates), mel_lens, tokens_lens




        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    conf = Tacotron2Config("config/configs/tacotron2_in_use.yaml")
    train_conf = Tacotron2Config("config/configs/train.yaml")
    ds = ljspeechDataset(conf, train_conf)
    dl = torch.utils.data.DataLoader(ds, batch_size=10, collate_fn=ds.collate_fn)
    t, m, g, l = next(iter(dl))

    print(l)
    plt.imshow(m[7])
    plt.show()
