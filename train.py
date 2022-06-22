import tensorflow as tf

from config.config import Tacotron2Config
from datasets.ljspeech import generate_map_func
from model.Tacotron2 import Tacotron2



if __name__ == "__main__":
    """
    initialize model
    """
    conf = Tacotron2Config("config/configs/tacotron2_laptop.yaml")
    tac = Tacotron2(conf)
    """
    initalize dataset
    """
    batch_size = conf["train"]["batch_size"]
    ljspeech_text = tf.data.TextLineDataset(conf["train_data"]["transcript_path"])
    tac.set_vocabulary(ljspeech_text.map(lambda x : tf.strings.split(x, sep='|')[1])) #initialize tokenizer and char. embedding
    map_F = generate_map_func(conf)
    ljspeech = ljspeech_text.map(map_F)
    """
    padding values :
        input : (phonem, mel spec), output : (mel spec, gate)
    """
    ljspeech = ljspeech.padded_batch(batch_size, 
            padding_values=((None, None, None), (None, 1.)),
            drop_remainder=conf["train"]["drop_remainder"])

    """
    train
    """
    optimizer = conf["train"]["optimizer"]
    epochs = conf["train"]["epochs"]

    tac.compile(optimizer=optimizer, loss=tac.criterion)
    tac.fit(ljspeech, epochs=epochs)
