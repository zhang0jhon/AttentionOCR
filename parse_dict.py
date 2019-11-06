#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: parse_dict.py

import os
import re
import numpy as np

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def get_dict(path=os.path.join(currentdir, 'label_dict/icdar_labels.txt'), add_space=False, add_eos=False):
    """
    Load text label dict from preprocessed text file.
    Args:
        path: label dict text file path.
        add_space: whether add additional space charater to label dict.
        add_eos: whether add EOS which represents end of sequence to label dict.
    Returns:
        label_dict: text label dict.
    """
    label_dict = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            m = re.match(r'(\d+) (.*)', line)
            idx, label = int(m.group(1)), m.group(2)
            label_dict[idx] = label 
        if add_space:
            label_dict[idx+1] = ' ' 
        if add_eos:
            label_dict[idx+2] = 'EOS' 
    return label_dict


if __name__ == '__main__':     
    label_dict = get_dict()
    print(label_dict)