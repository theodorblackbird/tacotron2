import tensorflow as tf
import time
from config.config import Tacotron2Config
from datasets.ljspeech import generate_map_func
from model.Tacotron2 import Tacotron2
from tqdm import tqdm


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
    ljspeech = ljspeech_text.map(map_F)
    """
    padding values :
        input : (phonem, mel spec), output : (mel spec, gate)
    """
    ljspeech = ljspeech.padded_batch(batch_size, 
            padding_values=((None, None, None), (None, 1.)),
            drop_remainder=conf["train"]["drop_remainder"])


    optimizer = conf["train"]["optimizer"]
    epochs = conf["train"]["epochs"]

    optimizer = tf.keras.optimizers.Adam()
    
    mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    mse_loss = tf.keras.losses.MeanSquaredError()
    bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for epoch in range(epochs):
        print(f"Epoch : {epoch}/{epochs}")
        start_time = time.time()
        for i, (x, y) in tqdm(enumerate(ljspeech), total=13100//batch_size):
            i += 1
            with tf.GradientTape() as tape:
                mels, gate = tac(x)
                mels, mels_post, mels_len = mels

                true_mels, true_gate = y

                crop = true_mels.shape[1] - true_mels.shape[1]%conf["n_frames_per_step"]#max_len must be a multiple of n_frames_per_step
                true_mels = true_mels[:,:crop,:]
                
                true_gate = true_gate[:,1:crop,:]
                true_gate = tf.reduce_mean(true_gate, axis = 2)
                one_slice = tf.expand_dims(tf.ones(true_gate.shape[0]), 1)
                true_gate = tf.concat([true_gate, one_slice], 1)
                true_gate = true_gate[:,::conf["n_frames_per_step"]]

                true_mels = tf.expand_dims(true_mels, -1)
                mels = tf.expand_dims(mels, -1)
                mels_post = tf.expand_dims(mels_post, -1)
                
                mels_mask = tf.sequence_mask(mels_len)

                loss = mse_loss(mels, true_mels) + \
                        mse_loss(mels_post, true_mels) + \
                        bce_logits(true_gate, gate) 

                grads = tape.gradient(loss, tac.trainable_weights)
                optimizer.apply_gradients(zip(grads, tac.trainable_weights))
            if i == 10:
                break
