#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import config as cfg
from tensorpack.dataflow import (
    DataFromList, MapData, MapDataComponent, RNGDataFlow, PrefetchData,
    MultiProcessMapData, MultiThreadMapData, TestDataSpeed, imgaug, BatchData
)

import cv2
import math
import numpy as np
import threading
import multiprocessing
from matplotlib import pyplot as plt
from contextlib import contextmanager

from common import (
    CustomResize, DataFromListOfDict, box_to_point8,
    filter_boxes_inside_shape, np_iou, point8_to_box, polygons_to_mask,
)

from dataset import LSVT, ART, ReCTS #, TotalText, ICDAR2017RCTW, MLT2019

def largest_size_at_most(height, width, largest_side, max_scale):
    """
    Compute resized image size with limited max scale. 
    """
    scale = largest_side/height if height>width else largest_side/width
    scale = min(scale, max_scale)

    new_height, new_width = height * scale, width * scale
    return new_height, new_width

def aspect_preserving_resize(image, largest_side, max_scale=4.):
    """
    Resize image with perserved aspect and limited max scale.
    """
    height, width = image.shape[:2]
    new_height, new_width = largest_size_at_most(height, width, largest_side, max_scale)

    new_height = max(new_height, cfg.stride)
    new_width = max(new_width, cfg.stride)
    resized_image = cv2.resize(image, (int(new_width), int(new_height)))

    return resized_image

def padding_image(image, padding_size):
    """
    Padding arbitrary-shaped text image to square for tensorflow batch training.
    """
    height, width = image.shape[:2]
    padding_h = padding_size - height
    padding_w = padding_size - width

    padding_top = np.random.randint(padding_h)
    padding_left = np.random.randint(padding_w)
    padding_down = padding_h - padding_top
    padding_right = padding_w - padding_left

    padding_img = cv2.copyMakeBorder(image, padding_top, padding_down, padding_left, padding_right, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    return padding_img, (padding_top, padding_left, height, width)


def rotatedPoint(R, point):
    """
    Transform polygon with affine transform matrix.
    """
    x = R[0,0]*point[0] + R[0,1]*point[1] + R[0,2]
    y = R[1,0]*point[0] + R[1,1]*point[1] + R[1,2]
    return [int(x), int(y)]


def affine_transform(image, polygon):    
    """
    Conduct same affine transform for both image and polygon for data augmentation.
    """ 
    height, width, _ = image.shape
    center_x, center_y = width/2, height/2

    angle = 0 if np.random.uniform()>0.5 else np.random.uniform(-20., 20.)
    shear_x, shear_y = (0,0) if np.random.uniform()>0.5 else (np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2))
  
    rad = math.radians(angle)
    sin, cos = math.sin(rad), math.cos(rad)   # x, y
    abs_sin, abs_cos = abs(sin), abs(cos)

    new_width = ((height * abs_sin) + (width * abs_cos))
    new_height = ((height * abs_cos) + (width * abs_sin))
    
    new_width += np.abs(shear_y*new_height)
    new_height += np.abs(shear_x*new_width)
    
    new_width = int(new_width)
    new_height = int(new_height)
    
    M = np.array([[cos, sin+shear_y,  new_width/2 - center_x + (1-cos)*center_x-(sin+shear_y)*center_y],
                          [-sin+shear_x, cos, new_height/2 - center_y + (sin-shear_x)*center_x+(1-cos)*center_y]])

    rotatedImage = cv2.warpAffine(image, M, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    height, width = rotatedImage.shape[:2]
    rotatedPoints = [rotatedPoint(M, point) for point in polygon]
    mask = polygons_to_mask([np.array(rotatedPoints, np.float32)], new_height, new_width)
    x, y, w, h = cv2.boundingRect(mask)
    mask = np.expand_dims(np.float32(mask), axis=-1)
    rotatedImage = rotatedImage * mask
    
    cropImage = rotatedImage[y:y+h, x:x+w,:]

    return cropImage


class TextDataPreprocessor:
    """
    Tensorpack text data preprocess function.
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, roidb):
        filename, label, mask, bbox, polygon = roidb['filename'], roidb['label'], roidb['mask'], roidb['bbox'], roidb['polygon'], 
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = affine_transform(img, polygon)
        # img = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] if image.shape[0]<cfg.stride/2 or image.shape[1]<cfg.stride/2 else image
        img = img if image.shape[0]<cfg.stride/2 or image.shape[1]<cfg.stride/2 else image

        largest_side = np.random.randint(cfg.crop_min_size, cfg.image_size)
        img = aspect_preserving_resize(img, largest_side)

        img, crop_bbox = padding_image(img, cfg.image_size)

        normalized_bbox = [coord/cfg.image_size for coord in crop_bbox]

        img = img.astype("float32")/255.


        ret = {"image": img, "label": label, "mask": mask, "normalized_bbox": normalized_bbox, "is_training":True, "dropout_keep_prob":0.5}

        return ret

def get_train_dataflow(roidb):
    """
    Tensorpack text dataflow.
    """
    ds = DataFromList(roidb, shuffle=True)
    preprocess = TextDataPreprocessor(cfg)

    buffer_size = cfg.num_threads * 10
    ds = MultiThreadMapData(ds, cfg.num_threads, preprocess, buffer_size=buffer_size)
    # ds = MultiProcessMapData(ds, cfg.num_workers, preprocess, buffer_size=buffer_size)
    ds = PrefetchData(ds, 100, multiprocessing.cpu_count() // 4)

    #ds = BatchData(ds, cfg.batch_size, remainder=True)

    return ds

def get_roidb(dataset_name):
    """
    Load generated numpy dataset for tensorpack dataflow.
    """
    dataset = np.load(dataset_name)[()]
    filenames, labels, masks, bboxes, points = dataset["filenames"], dataset["labels"], dataset["masks"], dataset["bboxes"], dataset["points"]

    roidb = []
    for filename, label, mask, bbox, polygon in zip(filenames, labels, masks, bboxes, points):
        item = {"filename":filename, "label":label, "mask":mask, "bbox":bbox, "polygon":polygon}
        roidb.append(item)
    return roidb



def get_batch_train_dataflow(roidbs, batch_size):
    """
    Tensorpack batch text dataflow.
    """
    batched_roidbs = []

    batch = []
    for i, d in enumerate(roidbs):
        if i % batch_size == 0:
            if len(batch) == batch_size:
                batched_roidbs.append(batch)
            batch = []
        batch.append(d)

    def preprocess(roidb_batch):
        """
        Tensorpack batch text data preprocess function.
        """
        datapoint_list = []
        for roidb in roidb_batch:
            filename, label, mask, bbox, polygon = roidb['filename'], roidb['label'], roidb['mask'], roidb['bbox'], roidb['polygon']
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


            image = affine_transform(img, polygon)
            # img = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :] if image.shape[0]<cfg.stride/2 or image.shape[1]<cfg.stride/2 else image
            img = img if image.shape[0]<cfg.stride/2 or image.shape[1]<cfg.stride/2 else image

            largest_side = np.random.randint(cfg.crop_min_size, cfg.image_size)
            img = aspect_preserving_resize(img, largest_side)

            img, crop_bbox = padding_image(img, cfg.image_size)

            normalized_bbox = [coord/cfg.image_size for coord in crop_bbox]

            img = img.astype("float32")/255.

            ret = {"image": img, "label": label, "mask": mask, "normalized_bbox": normalized_bbox}
            datapoint_list.append(ret)

        batched_datapoint = {"is_training":True, "dropout_keep_prob":0.5}
        for stackable_field in ["image", "label", "mask", "normalized_bbox"]:
            batched_datapoint[stackable_field] = np.stack([d[stackable_field] for d in datapoint_list])
        return batched_datapoint  

    ds = DataFromList(batched_roidbs, shuffle=True)
    ds = MultiThreadMapData(ds, cfg.num_threads, preprocess)
    # ds = PrefetchData(ds, 100, multiprocessing.cpu_count() // 4)
    return ds

if __name__ == "__main__":
    lsvt = LSVT()
    lsvt.load_data() 
    print(len(lsvt.filenames))

    filenames = lsvt.filenames
    labels = lsvt.labels
    masks = lsvt.masks
    bboxes = lsvt.bboxes
    points = lsvt.points

    roidb = []
    for filename, label, mask, bbox, polygon in zip(filenames, labels, masks, bboxes, points):
        item = {"filename":filename, "label":label, "mask":mask, "bbox":bbox, "polygon":polygon}
        roidb.append(item)

    ds = get_train_dataflow(roidb)

    from tensorpack.dataflow import PrintData
    ds = PrintData(ds, 10)
    # TestDataSpeed(ds, 50000).start()
    for k in ds:
        print(k['label'], k['mask'], k['normalized_bbox'])
        plt.imshow(k['image'])
        plt.show()