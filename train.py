
from config.config import Tacotron2Config

import os
import platform

from model.Tacotron2 import Tacotron2
"""
if platform.node() != "jean-zay3":
    import manage_gpus as gpl
"""
from datetime import datetime
from datasets.ljspeech import ljspeechDataset
import torch



if __name__ == "__main__":

    # silence verbose TF feedback
    if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    """
    if platform.node() != "jean-zay3":
        gpl.get_gpu_lock(gpu_device_id=2, soft=False)
    """

    """
    initialize model
    """
    date_now =  datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "logs/" + date_now


    conf = Tacotron2Config("config/configs/tacotron2_in_use.yaml")
    train_conf = Tacotron2Config("config/configs/train_in_use.yaml")
    model = Tacotron2(conf, train_conf)

    print("_______TRAIN HP_______")
    print(train_conf["train"])
    print("_______MODEL HP_______")
    print(conf.conf)

    """
    initalize dataset
    """
    batch_size = train_conf["train"]["batch_size"]

    ds = ljspeechDataset(conf, train_conf)
    model.set_vocabulary(len(ds.tokenizer.vocab))
    dl = torch.utils.data.DataLoader(
            ds, 
            batch_size=batch_size, 
            collate_fn=ds.collate_fn,
            )


    epochs = train_conf["train"]["epochs"]
    learning_rate = train_conf["train"]["lr"]

    for x in dl :
        model.train_step(x)




