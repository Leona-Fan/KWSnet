import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from model.kwsnet import kwsnet
from data.dataloader_kwsnet import S101DataModule
from sklearn.metrics import average_precision_score
import os

# Function to calculate Mean Average Precision (mAP)
def mean_average_precision(data_sorted):
    words = sorted(set([d[0] for d in data_sorted]))  
    map_all_words = []  

    for word in words:
        word_data = [d for d in data_sorted if d[0] == word]

        word_data = sorted(word_data, key=lambda x: -x[2])  
        labels = np.array([d[3] for d in word_data])  
        scores = np.array([d[2] for d in word_data]) 

        total_positives = np.sum(labels)  
        if total_positives == 0:  
            continue

        ap = average_precision_score(labels, scores)
        map_all_words.append(ap)
    
    # Return mean average precision (mAP) across all words
    mean_ap = np.mean(map_all_words) if map_all_words else 0
    return mean_ap

def main(args):
    model = kwsnet()
    checkpoint = torch.load(args.test_path)  
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    
    S101dataset = S101DataModule()

    trainer = Trainer(
        accelerator='gpu', devices=1, precision=16, max_epochs=1
    )
    
    print("Starting testing...")
    predictions = trainer.predict(model, datamodule=S101dataset)

    word_list, video_path, y_pred, y_true = [], [], [], []
    
    for batch in predictions:
        video_paths, word_lists, preds, trues = batch 
        word_list.extend(word_lists)
        video_path.extend(video_paths)
        y_pred.extend(preds)
        y_true.extend(trues)

    data = list(zip(word_list, video_path, y_pred, y_true))
    data_sorted = sorted(data, key=lambda x: (x[3], x[2])) 
    
    mean_ap = mean_average_precision(data_sorted)
    print("Mean Average Precision (mAP):", mean_ap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test and Calculate mAP')
    parser.add_argument('--test-path', type=str, required=True, help='Path to the checkpoint for testing')
    args = parser.parse_args()
    main(args)
