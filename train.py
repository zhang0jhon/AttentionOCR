#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import numpy as np

import tensorflow as tf

import time
import argparse

import config as cfg
from model.tensorpack_model import *
from text_dataflow import get_roidb, get_batch_train_dataflow

from tensorpack import *

os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in cfg.gpus])


def train():
    roidb = get_roidb(cfg.dataset_name)
    train_dataflow = get_batch_train_dataflow(roidb, cfg.batch_size)

    logger.set_logger_dir(cfg.summary_path, 'd')
    
    # Compute the training schedule from the number of GPUs ...
    warmup_schedule = [(0, cfg.learning_rate/100), (cfg.warmup_steps, cfg.learning_rate)]
    # lr_schedule = [(int(cfg.warmup_steps/cfg.steps_per_epoch+0.5), cfg.learning_rate),(cfg.num_epochs-150, cfg.learning_rate/10),(cfg.num_epochs-50, cfg.learning_rate/100)]

    # Create callbacks ...
    callbacks = [
        PeriodicCallback(
            ModelSaver(max_to_keep=20, keep_checkpoint_every_n_hours=1),
            every_k_epochs=20),
        # linear warmup
        ScheduledHyperParamSetter(
            'learning_rate', warmup_schedule, interp='linear', step_based=True),
        # ScheduledHyperParamSetter('learning_rate', lr_schedule),
        GPUMemoryTracker(),
        HostMemoryTracker(),
        ThroughputTracker(samples_per_step=cfg.num_gpus),
        EstimatedTimeLeft(median=True),
        SessionRunTimeout(60000),   # 1 minute timeout
        GPUUtilizationTracker()
    ]


    # session_init = SmartInit(cfg.pretrain_path, ignore_mismatch=True)
    if cfg.restore_path:
        session_init = SmartInit(cfg.restore_path, ignore_mismatch=True)
    else:
        session_init = SaverRestoreRelaxed(cfg.pretrain_path, ignore=['global_step:0'])# if cfg.pretrain_path else SmartInit(cfg.restore_path, ignore_mismatch=True)

    model = AttentionOCR()

    traincfg = TrainConfig(
        model=model,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=cfg.steps_per_epoch,
        max_epoch=cfg.num_epochs,
        session_init=session_init,
        starting_epoch=cfg.starting_epoch
    )

    trainer = SyncMultiGPUTrainerReplicated(cfg.num_gpus, average=False, mode='nccl')

    launch_train_with_config(traincfg, trainer)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCR')

    parser.add_argument('--mode', type=str, help='train or test', default='train')
    
    args = parser.parse_args()

    if args.mode == 'train':
        train()
            
            