import tensorflow as tf

from config.config import Tacotron2Config
from datasets.ljspeech import ljspeechDataset
from model.Tacotron2 import Tacotron2, Tacotron2Loss

import os
import platform
import tensorflow_addons as tfa
#if platform.node() != "jean-zay3":
#   import manage_gpus as gpl
from datetime import datetime
from pprint import pprint
import matplotlib.cm

class NoamLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.math.maximum(1., tf.cast(step, tf.float32))
        return self.initial_learning_rate * tf.math.sqrt(self.warmup_steps) * \
                tf.math.minimum(step * tf.math.pow(self.warmup_steps, -1.5), tf.math.pow(step, -0.5))


class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds):
        self.g_step = 0
        self.ds = ds
        self.cm = tf.constant(matplotlib.cm.get_cmap('viridis').colors, dtype=tf.float32)

    def on_train_batch_end(self, batch, logs=None):
        tf.summary.scalar("loss", logs["loss"], step=self.g_step)
        tf.summary.scalar("gate_loss", logs["gate_loss"], step=self.g_step)
        tf.summary.scalar("pre_loss", logs["pre_loss"], step=self.g_step)
        tf.summary.scalar("post_loss", logs["post_loss"], step=self.g_step)
        self.g_step += 1
        if self.g_step%self.model.train_config["train"]["plot_every_n_batches"] == 0:
            x, y = next(iter(self.ds))
            mels, gate, alignments = self.model(x, training=True)
            mels, mels_post, mels_len = mels


            true_mels, true_gate = y

            true_mels = tf.expand_dims(true_mels, -1)

            mels_mask = tf.sequence_mask(mels_len)
            gate_mask = tf.sequence_mask(mels_len/self.model.config["n_frames_per_step"])

            mels_mask = tf.expand_dims(mels_mask, -1)
            mels_mask = tf.expand_dims(mels_mask, -1)

            mels = tf.where(mels_mask, mels, [0.])
            mels_post = tf.where(mels_mask, mels_post, [0.])

            alignments = tf.expand_dims(alignments, -1)
            
            normalize = lambda x: (x - tf.math.reduce_min(x))/(tf.math.reduce_max(x) - tf.math.reduce_min(x))

            mels_image = tf.squeeze(normalize(tf.transpose(mels, [0,2,1,3]))*255)
            true_mels_image = tf.squeeze(normalize(tf.transpose(true_mels, [0,2,1,3]))*255)
            alignments_image = tf.squeeze(normalize(tf.transpose(alignments, [0,2,1,3]))*255)

            mels_image = tf.cast(tf.round(mels_image), dtype=tf.int32)
            true_mels_image = tf.cast(tf.round(true_mels_image), dtype=tf.int32)
            alignments_image = tf.cast(tf.round(alignments_image), dtype=tf.int32)

            mels_image = tf.gather(self.cm, mels_image)
            true_mels_image = tf.gather(self.cm, true_mels_image)
            alignments_image = tf.gather(self.cm, alignments_image)

            tf.summary.image("mels", mels_image, step=self.g_step)
            tf.summary.image("true_mels", true_mels_image, step=self.g_step)
            tf.summary.image("alignments", alignments_image, step=self.g_step)



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
    pprint(train_conf["train"])
    print("_______MODEL HP_______")
    pprint(conf.conf)

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
    eval_ljspeech =ljspeech.padded_batch(batch_size, 
            padding_values=((None, None, None), (None, 0.)),
            drop_remainder=train_conf["train"]["drop_remainder"])
    loss_callback = LossCallback(eval_ljspeech)

    ljspeech = ljspeech.padded_batch(batch_size, 
            padding_values=((None, None, None), (None, 0.)),
            drop_remainder=train_conf["train"]["drop_remainder"])

    epochs = train_conf["train"]["epochs"]
    learning_rate = train_conf["train"]["lr"]
    warmup_steps = train_conf["train"]["warmup_steps"]

    optimizer = tfa.optimizers.RectifiedAdam(
            learning_rate=NoamLR(learning_rate, warmup_steps),
            weight_decay=train_conf["train"]["weight_decay"],
            clipnorm=train_conf["train"]["clip_norm"],)

    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, 
            update_freq='batch')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=train_conf["data"]["checkpoint_path"] + date_now,
            monitor='loss',
            verbose=1,
            save_weights_only=True,
            save_best_only=True,)

    tac.compile(optimizer=optimizer,
            run_eagerly=train_conf["train"]["run_eagerly"])

    tac.fit(ljspeech,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tensorboard_callback,
                cp_callback,
                loss_callback],)

