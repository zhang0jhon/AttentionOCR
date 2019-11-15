#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

import config as cfg
from common import polygons_to_mask

from model.tensorpack_model import *

from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter

def cal_sim(str1, str2):
    """
    Normalized Edit Distance metric (1-N.E.D specifically)
    """
    m = len(str1) + 1
    n = len(str2) + 1
    matrix = np.zeros((m, n))
    for i in range(m):
        matrix[i][0] = i
        
    for j in range(n):
        matrix[0][j] = j

    for i in range(1, m):
        for j in range(1, n):
            if str1[i - 1] == str2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(matrix[i - 1][j - 1], min(matrix[i][j - 1], matrix[i - 1][j])) + 1
    
    lev = matrix[m-1][n-1]
    if (max(m-1,n-1)) == 0:
        sim = 1.0
    else:
        sim = 1.0-lev/(max(m-1,n-1))
    return sim


def preprocess(image, points, size=cfg.image_size):
    """
    Preprocess for test.
    Args:
        image: test image
        points: text polygon
        size: test image size
    """
    height, width = image.shape[:2]
    mask = polygons_to_mask([np.asarray(points, np.float32)], height, width)
    x, y, w, h = cv2.boundingRect(mask)
    mask = np.expand_dims(np.float32(mask), axis=-1)
    image = image * mask
    image = image[y:y+h, x:x+w,:]

    new_height, new_width = (size, int(w*size/h)) if h>w else (int(h*size/w), size)
    image = cv2.resize(image, (new_width, new_height))

    if new_height > new_width:
        padding_top, padding_down = 0, 0
        padding_left = (size - new_width)//2
        padding_right = size - padding_left - new_width
    else:
        padding_left, padding_right = 0, 0
        padding_top = (size - new_height)//2
        padding_down = size - padding_top - new_height

    image = cv2.copyMakeBorder(image, padding_top, padding_down, padding_left, padding_right, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

    image = image/255.
    return image


def label2str(preds, probs, label_dict, eos='EOS'):
    """
    Predicted sequence to string. 
    """
    results = []
    for idx in preds:
        if label_dict[idx] == eos:
            break
        results.append(label_dict[idx])

    probabilities = probs[:min(len(results)+1, cfg.seq_len+1)]
    return ''.join(results), probabilities

def eval(args, filenames, polygons, labels, label_dict=cfg.label_dict):
    Normalized_ED = 0.
    total_num = 0
    total_time = 0

    model = AttentionOCR()
    predcfg = PredictConfig(
        model=model,
        session_init=SmartInit(args.checkpoint_path),
        input_names=model.get_inferene_tensor_names()[0],
        output_names=model.get_inferene_tensor_names()[1])

    predictor = OfflinePredictor(predcfg)

    for filename, points, label in zip(filenames, polygons, labels):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess(image, points, cfg.image_size)

        before = time.time()
        preds, probs = predictor(np.expand_dims(image, axis=0), np.ones([1,cfg.seq_len+1], np.int32), False, 1.)
        after = time.time()

        total_time += after - before
        preds, probs = label2str(preds[0], probs[0], label_dict)
        print(label)
        print(preds, probs)

        sim = cal_sim(preds, label)

        total_num += 1
        Normalized_ED += sim

    print("total_num: %d, 1-N.E.D: %.4f, average time: %.4f" % (total_num, Normalized_ED/total_num, total_time/total_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR')
    parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow model', default='./checkpoint/model-10000')
    args = parser.parse_args()

    from dataset import ICDAR2017RCTW

    ICDAR2017RCTW = ICDAR2017RCTW()
    ICDAR2017RCTW.load_data() 
    print(len(ICDAR2017RCTW.filenames))

    eval(args, ICDAR2017RCTW.filenames, ICDAR2017RCTW.points, ICDAR2017RCTW.transcripts)