import tensorflow as tf

from config.config import Tacotron2Config
from datasets.ljspeech import ljspeechDataset
from model.Tacotron2 import Tacotron2, Tacotron2Loss

import os
import platform
#if platform.node() != "jean-zay3":
#   import manage_gpus as gpl
from datetime import datetime




if __name__ == "__main__":

    # silence verbose TF feedback
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

    #if platform.node() != "jean-zay3":
    #    gpl.get_gpu_lock(gpu_device_id=2, soft=False)

    """
    initialize model
    """
    date_now =  datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/" + date_now

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    conf = Tacotron2Config("config/configs/tacotron2_in_use.yaml")
    train_conf = Tacotron2Config("config/configs/train_in_use.yaml")
    tac = Tacotron2(conf, train_conf)

    print("_______TRAIN HP_______")
    print(train_conf["train"])
    print("_______MODEL HP_______")
    print(conf.conf)

    """
    initalize dataset
    """

    batch_size = train_conf["train"]["batch_size"]
    ljspeech_text = tf.data.TextLineDataset(train_conf["data"]["transcript_path"])
    tac.set_vocabulary(ljspeech_text.map(lambda x : tf.strings.split(x, sep='|')[1])) #initialize tokenizer and char. embedding
    dataset_mapper = ljspeechDataset(conf, train_conf)
    ljspeech = ljspeech_text.map(dataset_mapper)

    """
    padding values :
        input : (phonem, mel spec), output : (mel spec, gate)
    """

    ljspeech = ljspeech.padded_batch(batch_size, 
            padding_values=((None, None, None), (None, 0.)),
            drop_remainder=train_conf["train"]["drop_remainder"])

    epochs = train_conf["train"]["epochs"]
    learning_rate = train_conf["train"]["lr"]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
            clipnorm=train_conf["train"]["clip_norm"])

    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, 
            update_freq='batch')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=train_conf["data"]["checkpoint_path"] + date_now,
            verbose=1,
            save_weights_only=True)

    tac.compile(optimizer=optimizer,
            run_eagerly=train_conf["train"]["run_eagerly"])
    tac.fit(ljspeech,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tensorboard_callback,
                cp_callback],)

