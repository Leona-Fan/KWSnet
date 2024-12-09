<h1 align="center">MAVSR2025 Track2 Baseline</h1>

## Introduction

The repository is the baseline code for the MAVSR2025 Track2 competition, designed to accomplish a basic visual keyword spotting(KWS). The model architecture is as follows:

![architecture](./pic/image.png)

## Preparation

#### Install dependencies:

```Shell
pip install -r requirements.txt
```

#### Download dataset:
To access the CAS-VSR-S101 dataset, please scan the signed agreement (https://github.com/VIPL-Audio-Visual-Speech-Understanding/AVSU-VIPL/blob/master/CAS-VSR-S101-Release%20Agreement.pdf) and send it to lipreading@vipl.ict.ac.cn. Please note that the dataset is only available to universities and research institutes for research purposes only. Note that the agreement should be signed by a full-time staff member (usually your tutor). Sharing the dataset with others is not allowed under the terms of the agreement.

#### Preprocess dataset

Enter `data` for data processing and preparation:

Download CAS-VSR-S101 and place it in `data/CAS-VSR-S101_zip/lip_imgs_96` for processing:

```Shell
python zip2pkl_101.py
```

## Training
Run the program `main.py` to train the model:

```
python main.py
```

To monitor training progress:

```
tensorboard --logdir /path_to_your_logdir/
```
Configurations are in 'config.py". Please pay attention that you may need to modify it to make the program work as expected.

## Test
We select 2000 videos and 300 words in validation set for test.
```Shell
python test.py
```


## Results

The table below lists the results of the baseline model trained on CAS-VSR-S101, showing the performance on validation and test sets:

| mAP on val | mAP on test |
| :--------: | :---------: |
|   \  |   19%    |

## Contact

For questions or further information, please contact:

- Email: lipreading@vipl.ict.ac.cn
- Organization: Institute of Computing Technology, Chinese Academy of Sciences 
