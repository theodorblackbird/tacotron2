from datasets.ljspeech import ljspeechDataset
from config.config import Tacotron2Config
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

if __name__ == "__main__":

    conf = Tacotron2Config("config/configs/tacotron2_laptop.yaml")  
    ljspeech_text = tf.data.TextLineDataset(conf["train_data"]["transcript_path"])
    dataset_mapper = ljspeechDataset(conf)
    ljspeech = ljspeech_text.map(dataset_mapper)
    scaler = StandardScaler()
    ljspeech = ljspeech.padded_batch(64, 
            padding_values=((None, None, None), (None, 1.)),
            drop_remainder=conf["train"]["drop_remainder"])
    for x, _ in tqdm(ljspeech, total=13100//64):
        _ , mel, _ = x
        mel = tf.reshape(mel, [-1, conf["n_mel_channels"]]).numpy()
        scaler.partial_fit(mel)
    print("mean : ", scaler.mean_)
    print("var : ", scaler.var_)
   
