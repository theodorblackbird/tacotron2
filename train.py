import tensorflow as tf

from config.config import Tacotron2Config
from datasets.ljspeech import generate_map_func
from model.Tacotron2 import Tacotron2



if __name__ == "__main__":
    """
    initialize model
    """
    conf = Tacotron2Config("config/configs/tacotron2.yaml")
    tac = Tacotron2(conf)
    """
    initalize dataset
    """
    batch_size = conf["train"]["batch_size"]
    ljspeech_text = tf.data.TextLineDataset(conf["train_data"]["transcript_path"])
    tac.set_vocabulary(ljspeech_text.map(lambda x : tf.strings.split(x, sep='|')[1])) #initialize tokenizer and char. embedding
    map_F = generate_map_func(conf)
    ljspeech_text = ljspeech_text.map(map_F)
    """
    padding values :
        input : (phonem, mel spec), output : (mel spec, gate)
    """
    ljspeech_text = ljspeech_text.padded_batch(batch_size, 
            padding_values=((None, None), (None, 1.)) )

    print(next(iter(ljspeech_text))[0][0])
    """
    train
    """
    optimizer = conf["train"]["optimizer"]
    epochs = conf["train"]["epochs"]

    tac.compile(optimizer=optimizer, loss=tac.criterion)
    tac.fit(ljspeech_text, epochs=epochs)
