from torch.utils.tensorboard import SummaryWriter
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
from tqdm import tqdm
from matplotlib import pyplot as plt

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
    writer = SummaryWriter()
    


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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    step = 0
    for e in tqdm(range(epochs)):
        for batch in tqdm(dl) :
            optimizer.zero_grad()
            metrics = model.train_step(batch)
            metrics["loss"].backward()
            optimizer.step()
            writer.add_scalar("loss", metrics["loss"], step)
            writer.add_scalar("gate_loss", metrics["gate_loss"], step)
            if step%100 == 0:
                writer.add_image("alignment", metrics["alignments"][0], step, dataformats="HW")
                writer.add_image("mels", metrics["mels"].squeeze(-1)[0], step, dataformats="HW")
            print("loss", metrics["loss"])
            print("gate_loss", metrics["gate_loss"])
            print("pre_loss", metrics["pre_loss"])
            print("post_loss", metrics["post_loss"])
            step += 1


        torch.save(model.state_dict(), train_conf["data"]["checkpoint_path"]+"save_"+date_now+"_epoch_"+str(e))
