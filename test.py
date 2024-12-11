import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from model.kwsnet import kwsnet
from data.dataloader_kwsnet import S101DataModule
from sklearn.metrics import average_precision_score
import config as config
import os

def main(args):
    model = kwsnet()
    checkpoint = torch.load(args.test_path)  
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    S101dataset = S101DataModule()

    trainer = Trainer(
        accelerator='gpu', devices=4, precision=16, max_epochs=1
    )
    
    print("Starting testing...")
    predictions = trainer.predict(model, datamodule=S101dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test and Calculate mAP')
    parser.add_argument('--test-path', type=str, default = config.test_path, help='Path to the checkpoint for testing')
    args = parser.parse_args()
    main(args)
