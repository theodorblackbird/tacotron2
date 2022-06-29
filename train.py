import tensorflow as tf

from config.config import Tacotron2Config
from datasets.ljspeech import ljspeechDataset
from model.Tacotron2 import Tacotron2, Tacotron2Loss

import os
import manage_gpus as gpl
from datetime import datetime




if __name__ == "__main__":

    # silence verbose TF feedback
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

    gpl.get_gpu_lock(gpu_device_id=2, soft=False)

    """
    initialize model
    """

    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    conf = Tacotron2Config("config/configs/tacotron2.yaml")
    tac = Tacotron2(conf)
    train_conf = conf["train"]

    """
    initalize dataset
    """

    batch_size = train_conf["batch_size"]
    ljspeech_text = tf.data.TextLineDataset(conf["train_data"]["transcript_path"])
    tac.set_vocabulary(ljspeech_text.map(lambda x : tf.strings.split(x, sep='|')[1])) #initialize tokenizer and char. embedding
    dataset_mapper = ljspeechDataset(conf)
    ljspeech = ljspeech_text.map(dataset_mapper)

    """
    padding values :
        input : (phonem, mel spec), output : (mel spec, gate)
    """

    ljspeech = ljspeech.padded_batch(batch_size, 
            padding_values=((None, None, None), (None, 1.)),
            drop_remainder=train_conf["drop_remainder"])

    epochs = train_conf["epochs"]

    optimizer = tf.keras.optimizers.Adam()
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", 
            update_freq='batch')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=conf["train_data"]["checkpoint_path"],
            verbose=1,
            save_weights_only=True)

    tac.compile(optimizer=optimizer)
    tac.fit(ljspeech,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tensorboard_callback,
                cp_callback])

