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
        with open(self.transcript_path) as f:
            self.text = f.readlines()
        self.tokenizer = Tokenizer()
        self.melspec = MelSpec()
    
    def __len__(self):
        return len(self.transcript_path)

    def __getitem__(self, idx):
        line = self.text[idx]
        line = line.split("|")
        locution_id, locution = line
        wav_path = os.path.join(self.audio_path, locution_id+".wav")
        y, sr = librosa.load(wav_path)
        mel = self.melspec(y)
        locution = locution.strip()
        tokens = self.tokenizer.encode(locution)

        return tokens, mel


    def collate_fn(self, batch):
        max_tokens_len = max(map(lambda x: len(x[0]), batch))
        mel_lens = list(map(lambda x: x[1].shape[1], batch))
        max_mel_len = max(mel_lens)
        mels, tokens, gates = [], [], []
        for t, m in batch :
            padded_mel = torch.zeros(conf["mel_spec"]["n_mel_channels"], max_mel_len)
            padded_tokens = torch.zeros(max_tokens_len)
            padded_gate = torch.ones(max_mel_len)

            padded_mel[:, :m.shape[1]] = torch.from_numpy(m)
            padded_tokens[:len(t)] = torch.Tensor(t)
            padded_gate[:m.shape[1]-1] = 0

            mels.append(padded_mel)
            tokens.append(padded_tokens)
            gates.append(padded_gate)

        return torch.stack(tokens), torch.stack(mels), torch.stack(gates), mel_lens




        

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
