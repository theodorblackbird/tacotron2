from numpy.core.numeric import count_nonzero
import tensorflow as tf

from config.config import Tacotron2Config
from datasets.ljspeech import ljspeechDataset
from model.Tacotron2 import Tacotron2

from tqdm import tqdm
import time
from datetime import datetime
import os

import manage_gpus as gpl




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

    optimizer = train_conf["optimizer"]
    epochs = train_conf["epochs"]

    optimizer = tf.keras.optimizers.Adam()
    
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=tac)
    manager = tf.train.CheckpointManager(checkpoint, directory="/tmp/test", max_to_keep=2)
    n_step_per_epoch = 13100//batch_size

    for epoch in range(epochs):
        print(f"Epoch : {epoch}/{epochs}")
        start_time = time.time()
        
        for i, (x, y) in tqdm(enumerate(ljspeech), total=n_step_per_epoch):
            with tf.GradientTape() as tape:
                mels, gate = tac(x)
                mels, mels_post, mels_len = mels

                true_mels, true_gate = y

                crop = tf.shape(true_mels)[1] - tf.shape(true_mels)[1]%conf["n_frames_per_step"]#max_len must be a multiple of n_frames_per_step

                """
                compute loss
                """

                true_mels = true_mels[:,:crop,:] 
                true_mels = tf.expand_dims(true_mels, -1)
                
                mels_mask = tf.sequence_mask(mels_len)
                mels_mask = tf.expand_dims(mels_mask, -1)
                mels_mask = tf.expand_dims(mels_mask, -1)

                mels = tf.where(mels_mask, mels, [0.])
                mels_post = tf.where(mels_mask, mels_post, [0.])

                if i%100==0 :
                    tf.summary.image("true",tf.transpose(true_mels, perm=[0,2,1,3]), step=i + epoch*n_step_per_epoch)
                    tf.summary.image("pred", tf.transpose(mels, perm=[0,2,1,3]), step=i + epoch*n_step_per_epoch)

                mels_mask = tf.broadcast_to(mels_mask, tf.shape(mels))
                mels_fac = tf.math.count_nonzero(mels_mask, dtype=tf.float32)


                loss = (mse_loss(mels, true_mels) + mse_loss(mels_post, true_mels)) / mels_fac + bce_logits(true_gate, gate) 

                tf.summary.scalar('loss', loss, step=i)

                grads = tape.gradient(loss, tac.trainable_weights)
                optimizer.apply_gradients(zip(grads, tac.trainable_weights))

        manager.save()
