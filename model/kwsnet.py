import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
import pytorch_lightning as pl
from torch.nn import BatchNorm1d, Dropout2d, Embedding, LSTM, Linear, Module
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import optimizer
import config as config
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score
from scipy import interpolate
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score



class kwsnet(pl.LightningModule):
    def __init__(self, dropout_p = 0.2, d_v = 512, d_e = 128):
        super(kwsnet,self).__init__()
        self.d_v = d_v
        self.d_e = d_e

        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(
                1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(
                1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
        self.bilstm1 = LSTM(512, d_v, batch_first=True, bidirectional=True)
        self.encoder = nn.GRU(d_e, d_e // 2 , batch_first=True, bidirectional=True)
        self.bilstm2 = LSTM(d_v + d_e, d_v // 2, batch_first=True, bidirectional=True)
        self.Embedding = nn.Embedding(3994,d_e) #vocab_size, embedding_dim

        self.bn1 = BatchNorm1d(512)
        self.bn2 = BatchNorm1d(d_v + d_e)
        self.fc_1 = Linear(2 * d_v, d_v)
        self.fc1= nn.Linear(512, 256)
        self.fc2= nn.Linear(256, 128)
        self.fc3= nn.Linear(128, 1)
        self.a1 = nn.LeakyReLU()
        self.a2 = nn.LeakyReLU()

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)

        self.param_init()
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()


    def forward(self,input_):
        videos, words, num_words = input_ 
        seq_len = videos.shape[2]
        words = words[:sum(num_words), ...]

        x = self.frontend3D(videos)
        x = x.transpose(1, 2)
        B, T = x.size(0), x.size(1)
        x = x.contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.resnet18(x)
        x = x.view(B, T, -1)
        x = self.bn1(x.transpose(1, 2))
        x = self.dropout(x.unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        o, _ = self.bilstm1(x)
        x= self.fc_1(o)
        x = torch.cat([features.expand((num_words[i].item(), -1, -1)) for i, features in enumerate(x)]) #[num_words,T,d_v)]
        
        x_w = self.Embedding(words) 
        o_e, h_e = self.encoder(x_w) 
        flattened = h_e.transpose(0, 1).contiguous().view((-1, self.d_e))
        o_e = torch.unsqueeze(flattened, 1).expand((-1, seq_len, -1))
        
        y_x = torch.cat((x,o_e),dim=-1) 
        y_x = self.bn2(y_x.transpose(1, 2)).transpose(1, 2)
        x,_ = self.bilstm2(y_x)
        x = self.a1(self.fc1(x))
        x = torch.sum(x, -2).squeeze() 
        x = self.dropout(x)
        x = self.a2(self.fc2(x))
        x = self.fc3(x)
        
        return x 

    def training_step(self,train_batch,train_idx):      
        x,y = train_batch
        y_pred = self(x)
        loss = self.loss(y_pred,y)
        y_pred_sig = self.sigmoid(y_pred)
        binary_accuracy = y_pred_sig.round().eq(y).float().mean()
        self.log("lr",self.optimizers().param_groups[0]['lr'],on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_binary_accuracy", float(binary_accuracy), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return{'loss':loss,'pred':y_pred}

    def validation_step(self, val_batch, batch_idx):
        x,y = val_batch
        y_pred = self(x)
        loss = self.loss(y_pred,y)
        y_pred_sig = self.sigmoid(y_pred)
        binary_accuracy = y_pred_sig.round().eq(y).float().mean()
        self.log("lr",self.optimizers().param_groups[0]['lr'],on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_binary_accuracy", float(binary_accuracy), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return{'loss':loss,'pred':y_pred}

    def test_step(self, text_batch, batch_idx):
        input_,y = text_batch
        videos, grapheme_tensor, num_words,video_path,word_list = input_
        new_input = (videos,grapheme_tensor,num_words)
        y_pred = self(new_input)
        loss = self.loss(y_pred,y)
        y_pred_sig = self.sigmoid(y_pred)
        binary_accuracy = y_pred_sig.round().eq(y).float().mean()
        self.log("test_binary_accuracy", float(binary_accuracy), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    
    def predict_step(self,text_batch, batch_idx):
        self.eval() #forbid grad
        input_ = text_batch
        videos, grapheme_tensor, num_words,video_path,word_list = input_
        new_input = (videos,grapheme_tensor,num_words)
        y_pred = self(new_input)
        y_pred_sig = self.sigmoid(y_pred)
        with open(f'analyze/record_{config.name}.txt', 'a') as f:
            for word,video,pred,true in zip(word_list,video_path,y_pred_sig,y):
                pred_value = pred.item()
                true_value = true.item()
                f.write(f"{word}\t{video}\t{pred_value:.16f}\t\n")
        return video_path,word_list,y_pred_sig


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=config.lr,
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,step_size = 10, gamma=0.5)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'epoch', 
            }
        }
    
    def param_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()  

class ResNet(nn.Module):
    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x) # [B*T,512,3,3]->[5800,512,1,1]
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x        

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se
        
        if(self.se):
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes//16)
            self.conv4 = conv1x1(planes//16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        if(self.se):
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()
            
            out = out * w
        
        out = out + residual
        out = self.relu(out)

        return out
