import argparse
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import config as config
from pytorch_lightning import Trainer, callbacks, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model.kwsnet import kwsnet
from data.dataloader_kwsnet import S101DataModule, S101dataset
from pytorch_lightning.strategies import DDPStrategy
from datetime import datetime
import os
import random
    
def main(args):
    model = kwsnet()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger(args.model+"_logs", name=current_time, version=config.version)
    model_checkpoint_callback=ModelCheckpoint(monitor='val_binary_accuracy',filename="checkpoints-{epoch:02d}-{train_loss:.2f}-{train_binary_accuracy:.2f}-{val_loss:.2f}-{val_binary_accuracy:.2f}",save_top_k=-1)

    trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=True),
                      logger=logger,
                      accelerator='gpu',
                      devices=config.gpus,
                      callbacks=[model_checkpoint_callback],
                      precision="16-mixed",
                      max_epochs=config.max_epoch)
    checkpoint =torch.load(config.pre_train_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict,strict = False)

    #dataset
    S101dataset = S101DataModule()
    trainer.fit(model,S101dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kwsnet on 101')
    parser.add_argument('--model', type=str, default='kwsnet', help='model')
    args = parser.parse_args()

    main(args)
