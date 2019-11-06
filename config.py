# -*- coding: utf-8 -*-

import os
from parse_dict import get_dict

# base dir for multiple text datasets
base_dir = '/opt/data/nfs/zhangjinjin/data/text/'

# font path for visualization
font_path = './fonts/cn/SourceHanSans-Normal.ttf'

# 'ocr' for inception model with padding image. 
# 'ocr_with_normalized_bbox' for inception model with cropped text region for attention lstm.
model_name = 'ocr' # 'ocr_with_normalized_bbox'

# path for tensorboard summary and checkpoint path 
summary_path = './checkpoint'

# tensorflow model name scope
name_scope = 'InceptionV4'

# path for numpy dict with processed image paths and labels used in dataset.py
dataset_name = 'icdar_datasets.npy'
# pb_path = './checkpoint/text_recognition_5435.pb'

# restore training parameters
restore_path = ''
starting_epoch = 1

# checkpoint_path = './checkpoint/model-10000'
# imagenet pretrain model path 
pretrain_path = './pretrain/inception_v4.ckpt'

# label dict for text recognition
label_dict = get_dict()
reverse_label_dict = dict((v,k) for k,v in label_dict.items())

# gpu lists
gpus = [6, 7, 8, 9]

num_gpus = len(gpus)
num_classes = len(label_dict)

# max sequence length without EOS 
seq_len = 32

# embedding size
wemb_size = 256

# lstm size
lstm_size = 512

# minimum cropped image size for data augment
crop_min_size = 224

# input image size
image_size = 256

# max random image offset for data augment
offset = 16

# CNN endpoint stride
stride = 8

# resize parameters for data augment
TRAIN_SHORT_EDGE_SIZE = 8
MAX_SIZE = image_size - 32

# training batch size
batch_size = 10 #12

# steps per training epoch in tensorpack
steps_per_epoch = 500

# max epochs 
num_epochs = 1000

# model weight decay factor
weight_decay = 1e-5

# base learning rate
learning_rate = 1e-4 * num_gpus

# minimun learning rate for cosine decay learning rate
min_lr = learning_rate / 100

# warm up steps 
warmup_steps = 1000

# thread for multi-thread data loading  
num_threads = 16
