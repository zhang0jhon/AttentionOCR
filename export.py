#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import config as cfg

from model.tensorpack_model import *

from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter

def export(args):
    model = AttentionOCR()
    predcfg = PredictConfig(
        model=model,
        session_init=SmartInit(args.checkpoint_path),
        input_names=model.get_inferene_tensor_names()[0],
        output_names=model.get_inferene_tensor_names()[1])

    ModelExporter(predcfg).export_compact(args.pb_path, optimize=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR')

    parser.add_argument('--pb_path', type=str, help='path to save tensorflow pb model', default='./checkpoint/text_recognition_5435.pb')
    parser.add_argument('--checkpoint_path', type=str, help='path to tensorflow model', default='./checkpoint/model-10000')

    args = parser.parse_args()
    export(args)