#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

import cv2
from matplotlib import pyplot as plt
import json
import codecs


from PIL import Image, ImageDraw, ImageFont 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import config as cfg

max_len = cfg.seq_len + 1
base_dir = cfg.base_dir
font_path = cfg.font_path

dataset_path = {  'art': os.path.join(base_dir, 'art/train_task2_images'), 
                  'rects': os.path.join(base_dir, 'rects/img'),
                  'lsvt': os.path.join(base_dir, 'lsvt/train'),
                  'icdar2017rctw': os.path.join(base_dir, 'icdar2017rctw/train'), } 

lsvt_annotation = os.path.join(base_dir, 'lsvt/train_full_labels.json')
art_annotation = os.path.join(base_dir, 'art/train_task2_labels.json')



def visualization(image_path, points, label, vis_color = (255,255,255)):
    """
    Visualize groundtruth label to image.
    """
    points = np.asarray(points, dtype=np.int32)
    points = np.reshape(points, [-1,2])
    image = cv2.imread(image_path)
    cv2.polylines(image, [points], 1, (0,255,0), 2)
    image = Image.fromarray(image)
    FONT = ImageFont.truetype(font_path, 20, encoding='utf-8')   
    DRAW = ImageDraw.Draw(image)  
    
    DRAW.text(points[0], label, vis_color, font=FONT)
    return np.array(image)

def strQ2B(uchar):
    """
    Convert full-width character to half-width character.
    """
    inside_code = ord(uchar)
    if inside_code == 12288:
        inside_code = 32
    elif (inside_code >= 65281 and inside_code <= 65374):
        inside_code -= 65248
    return chr(inside_code)

def preprocess(string):
    """
    Groundtruth label preprocess function.
    """
    # string = [strQ2B(ch) for ch in string.strip()]
    # return ''.join(string)  
    return string  


class Dataset(object):
    """
    Base class for text dataset preprocess.
    """
    def __init__(self, name='base', max_len=max_len, base_dir=base_dir, label_dict=cfg.reverse_label_dict): # label_dict  label_dict_with_rects 5434+1
        self.data_path = dataset_path[name]
        print(self.data_path)
        self.label_dict = label_dict
        self.max_len = max_len
        self.base_dir = base_dir
        self.filenames = []
        self.labels = []
        self.masks = []
        self.bboxes = []
        self.points = []

                       
                    
class ReCTS(Dataset):
    """
    ICDAR2019 ReCTS dataset, refer to https://rrc.cvc.uab.es/?ch=12&com=downloads.
    """
    def __init__(self, name='rects'):
        super(ReCTS, self).__init__(name=name)

    def load_data(self):
        label_folder = os.path.join(self.base_dir, 'rects/gt_unicode/') #gt_unicode gt
        
        for filename in os.listdir(label_folder):
            img_name = os.path.join(self.data_path, filename[:-5]+'.jpg')
            # image = cv2.imread(img_name)
            # print(img_name)
            with open(os.path.join(label_folder, filename)) as f:
                json_data = json.load(f)
                anno_data = json_data['lines']
                points = [anno['points'] for anno in anno_data]
                transcripts = [anno['transcription'] for anno in anno_data]
                ignores = [anno['ignore'] for anno in anno_data]
                for polygon, transcript, ignore in zip(points, transcripts, ignores):
                    if ignore:
                        continue
                        
                    if len(transcript)>self.max_len-1:
                        continue
                    
                    if transcript=='###':
                        continue
                    
                    transcript = preprocess(transcript)  

                    skip = False
                    for char in transcript:
                        if char not in self.label_dict.keys():
                            skip = True
    
                    if skip:
                        print(transcript)
                        continue
                        
                    seq_label = []     
                    for char in transcript:
                        seq_label.append(self.label_dict[char])#.decode('utf-8')
                    seq_label.append(self.label_dict['EOS'])
                    
                    non_zero_count = len(seq_label)
                    seq_label = seq_label + [self.label_dict['EOS']]*(self.max_len-non_zero_count)
                    mask = [1]*(non_zero_count) + [0]*(self.max_len-non_zero_count) 
                    
                    polygon = np.array(polygon, dtype=np.int64)   
                    polygon = np.reshape(polygon, (-1,2))
                    
                    points_x = [point[0] for point in polygon]
                    points_y = [point[1] for point in polygon]
                    bbox = [np.amin(points_y), np.amin(points_x), np.amax(points_y), np.amax(points_x)] # ymin, xmin, ymax, xmax
                    bbox = [int(item) for item in bbox]

                    bbox_w, bbox_h = bbox[3]-bbox[1], bbox[2]-bbox[0]
                    if bbox_w <8 or bbox_h <8:
                        continue

                    # print(transcript, seq_label, mask, polygon)
                    # img = visualization(img_name, polygon, transcript)
                    # plt.imshow(img)
                    # plt.show()

                    self.filenames.append(img_name)
                    self.labels.append(seq_label)
                    self.masks.append(mask)
                    self.bboxes.append(bbox)
                    self.points.append(polygon)

         
        
class ART(Dataset):
    """
    ICDAR2019 ArT dataset, refer to https://rrc.cvc.uab.es/?ch=14&com=downloads.
    """
    def __init__(self, name='art'):
        super(ART, self).__init__(name=name)

    def load_data(self, annotation_file=art_annotation):
        count = 0
        with open(annotation_file) as f:
            json_data = json.load(f)

            for filename in os.listdir(self.data_path):
                img_name = os.path.join(self.data_path, filename)
                #image = cv2.imread(img_name)
                #image_height, image_width = image.shape[:2]

                anno_data = json_data[filename[:-4]][0]
                # print(len(json_data[filename[:-4]]))
                illegibility = anno_data['illegibility']

                if illegibility:
                    continue

                polygon = anno_data['points']
                transcripts = anno_data['transcription']
                languages = anno_data['language']

                if len(transcripts)>self.max_len-1:
                    # print(transcripts)
                    # count = count + 1
                    continue

                transcripts = preprocess(transcripts) 

                skip = False
                for char in transcripts:
                    if char not in self.label_dict.keys():
                        skip = True

                if skip:
                    # print(transcripts)
                    count = count + 1
                    continue

                # print(polygon, transcripts)

                seq_label = []     
                for char in transcripts:
                    seq_label.append(self.label_dict[char])#.decode('utf-8')
                seq_label.append(self.label_dict['EOS'])
                
                non_zero_count = len(seq_label)
                seq_label = seq_label + [self.label_dict['EOS']]*(self.max_len-non_zero_count)
                mask = [1]*(non_zero_count) + [0]*(self.max_len-non_zero_count) 

                points_x = [point[0] for point in polygon]
                points_y = [point[1] for point in polygon]
                bbox = [np.amin(points_y), np.amin(points_x), np.amax(points_y), np.amax(points_x)] # ymin, xmin, ymax, xmax
                bbox = [int(item) for item in bbox]
                
                bbox_w, bbox_h = bbox[3]-bbox[1], bbox[2]-bbox[0]

                if bbox_w <8 or bbox_h <8:
                    continue

                # print(transcripts, seq_label, mask, polygon)
                # img = visualization(img_name, polygon, transcripts)
                # plt.imshow(img)
                # plt.show()  

                self.filenames.append(img_name)
                self.labels.append(seq_label)
                self.masks.append(mask)
                self.bboxes.append(bbox)
                self.points.append(polygon)

        

class LSVT(Dataset):
    """
    ICDAR2019 LSVT dataset, refer to https://rrc.cvc.uab.es/?ch=16&com=downloads.
    """
    def __init__(self, name='lsvt'):
        super(LSVT, self).__init__(name=name)

    def load_data(self, annotation_file=lsvt_annotation):
        with open(annotation_file) as f:
            json_data = json.load(f)

            for filename in os.listdir(self.data_path):
                img_name = os.path.join(self.data_path, filename)
                #image = cv2.imread(img_name)
                #image_height, image_width = image.shape[:2]

                anno_data = json_data[filename[:-4]]
                # print(len(json_data[filename[:-4]]))
                # print(anno_data)
                points = [anno['points'] for anno in anno_data]
                transcripts = [anno['transcription'] for anno in anno_data]
                illegibilities = [anno['illegibility'] for anno in anno_data]

                for polygon, transcript, illegibility in zip(points, transcripts, illegibilities):
                    if transcript == '###':
                        continue

                    transcript = preprocess(transcript.strip())
                

                    if len(transcript)>self.max_len-1:
                        # print(transcripts)
                        # count = count + 1
                        continue

                    skip = False
                    for char in transcript:
                        if char not in self.label_dict.keys():
                            skip = True

                    if skip:
                        continue

                    # print(polygon, transcripts)

                    seq_label = []     
                    for char in transcript:
                        seq_label.append(self.label_dict[char])#.decode('utf-8')
                    seq_label.append(self.label_dict['EOS'])
                    
                    non_zero_count = len(seq_label)
                    seq_label = seq_label + [self.label_dict['EOS']]*(self.max_len-non_zero_count)
                    mask = [1]*(non_zero_count) + [0]*(self.max_len-non_zero_count) 

                    points_x = [point[0] for point in polygon]
                    points_y = [point[1] for point in polygon]
                    bbox = [np.amin(points_y), np.amin(points_x), np.amax(points_y), np.amax(points_x)] # ymin, xmin, ymax, xmax
                    bbox = [int(item) for item in bbox]
                    
                    bbox_w, bbox_h = bbox[3]-bbox[1], bbox[2]-bbox[0]

                    if bbox_w <8 or bbox_h <8:
                        continue

                    # print(transcript, seq_label, mask, polygon)
                    # img = visualization(img_name, polygon, transcript)
                    # plt.imshow(img)
                    # plt.show()  
                    self.filenames.append(img_name)
                    self.labels.append(seq_label)
                    self.masks.append(mask)
                    self.bboxes.append(bbox)
                    self.points.append(polygon)


class ICDAR2017RCTW(Dataset):
    """
    ICDAR2017 RCTW-17 dataset, refer to http://rctw.vlrlab.net/dataset/.
    """
    def __init__(self, name='icdar2017rctw'):
        super(ICDAR2017RCTW, self).__init__(name=name)
        self.transcripts = []
        
    def load_data(self):
        for filename in os.listdir(self.data_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(self.data_path, filename)
                with codecs.open(os.path.join(self.data_path, filename[:-4]+'.txt'), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        res = line.split(",", 10)
                        label = res[9][1:-2]#.decode('utf-8')
                        
                        if label=='###':
                            continue
                            
                        if len(label)>self.max_len-1:
                            continue
                        
                        label = preprocess(label) 

                        skip = False
                        for char in label:
                            if char not in self.label_dict.keys():
                                skip = True
                        #if label[0] not in label_dict.keys():
                        if skip:
                            continue
                            
                        seq_label = []     
                        for char in label:
                            seq_label.append(self.label_dict[char])#.decode('utf-8')
                        seq_label.append(self.label_dict['EOS'])
                        
                        non_zero_count = len(seq_label)
                        seq_label = seq_label + [self.label_dict['EOS']]*(self.max_len-non_zero_count)
                        mask = [1]*(non_zero_count) + [0]*(self.max_len-non_zero_count) 
                        try:
                            vertex_row_coords= [int(res[1]), int(res[3]), int(res[5]), int(res[7])]
                            vertex_col_coords = [int(res[0]), int(res[2]), int(res[4]), int(res[6])]
                        except:
                            continue
                                 
                        bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords), np.amax(vertex_col_coords)]
                        polygon = [[int(res[0]),int(res[1])],[int(res[2]),int(res[3])],[int(res[4]),int(res[5])],[int(res[6]),int(res[7])]]

                        #print(bbox[2]-bbox[0], bbox[3]-bbox[1])
                        bbox_w, bbox_h = bbox[3]-bbox[1], bbox[2]-bbox[0]
                        if bbox_w <8 or bbox_h <8:
                            continue

                        # print(polygon, label, seq_label, mask)
                        # image = visualization(img_path, polygon, label)
                        # plt.imshow(image)
                        # plt.show()

                        self.filenames.append(img_path)
                        self.labels.append(seq_label)
                        self.masks.append(mask)
                        self.bboxes.append(bbox)
                        self.points.append(polygon)
                        self.transcripts.append(label)

if __name__=='__main__':
    LSVT = LSVT()
    LSVT.load_data() 
    print(len(LSVT.filenames))

    ART = ART()
    ART.load_data()
    print(len(ART.filenames))

    ReCTS = ReCTS()
    ReCTS.load_data()
    print(len(ReCTS.filenames))
    
    filenames = LSVT.filenames + ART.filenames + ReCTS.filenames
    labels = LSVT.labels + ART.labels + ReCTS.labels
    masks = LSVT.masks + ART.masks + ReCTS.masks
    bboxes = LSVT.bboxes + ART.bboxes + ReCTS.bboxes
    points = LSVT.points + ART.points + ReCTS.points

    from sklearn.utils import shuffle
    filenames, labels, masks, bboxes, points = shuffle(filenames, labels, masks, bboxes, points, random_state=0)
    print(len(filenames))

    dataset = {"filenames":filenames, "labels":labels, "masks":masks, "bboxes":bboxes, "points":points}
    np.save(cfg.dataset_name, dataset)

    
    


    
    
    
    