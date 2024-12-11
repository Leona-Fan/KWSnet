import os
import torch
import io
import numpy as np
import cv2
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset
from turbojpeg import TurboJPEG
import config as config
from torchvision.transforms import Compose, CenterCrop, RandomCrop, Resize,Lambda, ToTensor, Grayscale
from torchvision.transforms.functional import to_pil_image
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
import random
from torch import tensor
import re
from kornia.augmentation import RandomGaussianNoise, RandomMotionBlur, ColorJitter, RandomRotation, RandomHorizontalFlip

word_dict = 'data/101/char_frequencies.txt'
char_dict = []
with open(word_dict, 'r', encoding='utf-8') as file:
    char_dict = [line.strip() for line in file if line.strip()]

class S101dataset(Dataset):
    def __init__(self, path=config.path, set="train", list_path='data/101/longform_transcripts.csv', augment=False):
        super().__init__()
        list_path = list_path.format(set)
        if set == "test":
            list_path = config.test_label_path
            with open(list_path) as f:
                self.list = [line.strip().split(',') for line in f.readlines()]
        else:
            with open(list_path) as f:
                self.list = [line.strip().split('\t') for line in f.readlines()]
        self.set = set
        self.augment = augment
        self.path = path
        if set == "train":
            self.list = [(video, 0, int(end), list(re.sub(r'[^\w\s]','',words)), i) for i, (video, start_second, end_second, duration, start, end, frames, words) in enumerate(self.list) if frames>0 and len(list(re.sub(r'[^\w\s]','',words)))>4]
        elif set == "val":
            self.list = [(video, 0, int(end), list(re.sub(r'[^\w\s]','',words)), i) for i, (video, start_second, end_second, duration, start, end, frames, words) in enumerate(self.list) if frames>0 and len(list(re.sub(r'[^\w\s]','',words)))>4]
        elif set == "test":
            self.list = [(video, 0, int(end), words, i) for i, (video, start, end, words) in enumerate(self.list)]
        

    def __getitem__(self,index): 
        video, start, end, words, i = self.list[index]
        video_path = os.path.join(self.path, video + ".pkl")
        video = np.load(video_path,allow_pickle=True)
        length = len(video)
        video = video[start:min(end,length)]

        if self.set == "train":
            video = [Image.open(io.BytesIO(frame)) for frame in video] 
            transform =  Compose([Grayscale(num_output_channels=1),Resize((96, 96)), RandomCrop((88, 88))])
            video = [transform(frame) for frame in video]
            if len(video) == 0:
                print(video_path,start,end, length, words)
            video = torch.stack([ToTensor()(frame) for frame in video])
        else:
            video = [Image.open(io.BytesIO(frame)) for frame in video]
            transform =  Compose([Grayscale(num_output_channels=1),Resize((96, 96)), CenterCrop((88, 88))])
            video = [transform(frame) for frame in video]
            if len(video) == 0:
                print(video_path,start,end, length, words)
            video = torch.stack([ToTensor()(frame) for frame in video])
        video = video.permute(1,0,2,3)
        if self.augment:
            imgs = video  
            imgs = RandomRotation(degrees=0.1, p=1.)(imgs)
            imgs = RandomGaussianNoise(mean=0., std=0.1, p=0.2)(imgs)
            imgs = RandomMotionBlur(kernel_size=(3, 7), angle=(0., 360.), direction=(-1., 1.), border_type='reflect', p=0.2)(imgs)
            imgs = RandomHorizontalFlip(p=0.5)(imgs)
            video = imgs
        if self.set == "test":
            return torch.FloatTensor(np.ascontiguousarray(video)), words,video_path, i
        return torch.FloatTensor(np.ascontiguousarray(video)), words
        
    def __len__(self):
        return len(self.list)

def make_negative_words(word_lists, negative_multiplier = 1): 
    word_pool = [word for sublist in word_lists for word in sublist]
    labels = []
    for words in word_lists:
        n = len(words)
        labels += [True] * n
        for k in range(negative_multiplier):
            labels += [False] * n
            for j in range(n):
                word = random.choice(word_pool) 
                loops = 0
                while word in words[:n]:
                    word = random.choice(word_pool)
                    loops += 1
                    if loops >= 1000:
                        raise LookupError("Cannot find negative word from", word_pool, ", differing from", words[:n])
                words.append(word)
    num_words = [len(words) for words in word_lists]
    word_list = sum(word_lists,[])
    
    return word_list, labels, num_words

def collate_fn(batch,test = config.test):
    if test:
        videos, word_lists,video_path,i= zip(*batch)
        num_words = [1] * len(word_lists)
        word_list = word_lists
    else:
        videos, word_lists= zip(*batch)
        words = []
        for word_list in word_lists:
            temp = []
            options = [4, 5]  
            weights = [9, 1] 
            for i in range(len(word_list) - 4): 
                chosen_length = random.choices(options, weights)[0]
                if i + chosen_length <= len(word_list):  
                    word = ''.join(word_list[i:i + chosen_length])
                    temp.append(word)
            words.append(temp)
        word_list, labels, num_words = make_negative_words(words)
    num_words = tensor(num_words, dtype=torch.long) 
    word_grapheme=[[char_dict.index(char) if char in char_dict else char_dict.index("<unk>") for char in word] for word in word_list]
    pad_index = char_dict.index('<pad>')
    max_length = max(len(word) for word in word_grapheme)
    padded_grapheme = [word + [pad_index] * (max_length - len(word)) for word in word_grapheme]
    grapheme_tensor = torch.tensor(padded_grapheme, dtype=torch.long)
    videos = pad_sequence([video.transpose(0, 1) for video in videos], batch_first=True, padding_value=0).transpose(1, 2) #batch_size,1,25,88,88
    if test:
        return videos,grapheme_tensor,num_words,video_path,word_list
    labels = tensor(labels, dtype=torch.float16).unsqueeze(-1)
    y = labels
    return (videos,grapheme_tensor,num_words), y

class S101DataModule(pl.LightningDataModule):
    def __init__(self, ):
        super().__init__()
        self.is_distributed = config.gpus > 1

    def train_dataloader(self):
        train_dataset = S101dataset(path=config.path,set='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True,collate_fn=collate_fn)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = S101dataset(path=config.path,set='val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=False,collate_fn=collate_fn)
        return val_loader

    def test_dataloader(self):
        test_dataset = S101dataset(path=config.path,set='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=False,collate_fn=collate_fn)
        return test_loader
    
    def predict_dataloader(self):
        test_dataset = S101dataset(path=config.path,set='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=False,collate_fn=collate_fn)
        return test_loader

