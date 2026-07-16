# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:34:27 2020

@author: acer
"""

import os
import numpy as np
import SimpleITK as sitk
from functools import cmp_to_key
from .Data_augment_3D import composite_transform, apply_transform

def load_nifti_image(filepath):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    # pathname = filepath.split('\\')
    # name = pathname[-2]
    data_itk = sitk.ReadImage(filepath)

    properties = dict()
    properties["itk_origin"] = data_itk.GetOrigin()
    properties["itk_spacing"] = data_itk.GetSpacing()
    properties["itk_direction"] = data_itk.GetDirection()

    return data_itk,properties

def cmpfun(x, y):
    xs = x.split('.')[0].split('_')
    ys = y.split('.')[0].split('_')
    if int(xs[0]) < int(ys[0]):
        return -1
    elif int(xs[0]) > int(ys[0]):
        return 1
    elif int(xs[0]) == int(ys[0]):
        if int(xs[1]) < int(ys[1]):
            return -1
        elif int(xs[1]) > int(ys[1]):
            return 1
        else:
            return 0

def augmentation(path_img,path_label,aug=False):
    imgs = [load_nifti_image(path)[0] for path in path_img]
    labels = [load_nifti_image(path)[0] for path in path_label]
    
    i=0
    res = 96

    # translate, scale, flip, rotate, deform
    if aug:
        if np.random.rand() > 0.1:
            t = composite_transform(imgs[-1], probs = (1.0, 1.0, 1.0, 1.0, 0.0))
            imgs = [apply_transform(img, t) for img in imgs]
            labels = [apply_transform(label, t) for label in labels]
    
    properties = dict()
    properties["itk_origin"] = imgs[-1].GetOrigin()
    properties["itk_spacing"] = imgs[-1].GetSpacing()
    properties["itk_direction"] = imgs[-1].GetDirection()
    data_npys = [sitk.GetArrayFromImage(img) for img in imgs]
    label_npys = [sitk.GetArrayFromImage(label) for label in labels]
    
    a, b, c = np.where((label_npys[-1] == 1))
    am = label_npys[-1].shape[0]
    bm = label_npys[-1].shape[1]
    cm = label_npys[-1].shape[2]
    pada = 0
    padb = 0
    padc = 0
    if am < res:
        a1 = 0
        a2 = am
        pada = res - am
    else:
        a1 = (np.max(a) + np.min(a)) // 2 - res//2
        a2 = (np.max(a) + np.min(a)) // 2 + res//2
        if a1 > np.min(a):
            a1 = np.min(a)
            a2 = a1 + res
        if a2 < np.max(a):
            a2 = np.max(a)
            a1 = a2 - res
        if a1 < 0:
            a1 = 0
            a2 = res
        elif a2 > am:
            a1 = am-res
            a2 = am
    if bm < res:
        b1 = 0
        b2 = bm
        padb = res - bm
    else:
        b1 = (np.max(b) + np.min(b)) // 2 - res//2
        b2 = (np.max(b) + np.min(b)) // 2 + res//2
        if b1 > np.min(b):
            b1 = np.min(b)
            b2 = b1 + res
        if b2 < np.max(b):
            b2 = np.max(b)
            b1 = b2 - res
        if b1 < 0:
            b1 = 0
            b2 = res
        elif b2 > bm:
            b1 = bm-res
            b2 = bm
    if cm < res:
        c1 = 0
        c2 = cm
        padc = res - cm
    else:
        c1 = (np.max(c) + np.min(c)) // 2 - res//2
        c2 = (np.max(c) + np.min(c)) // 2 + res//2
        if c1 > np.min(c):
            c1 = np.min(c)
            c2 = c1 + res
        if c2 < np.max(c):
            c2 = np.max(c)
            c1 = c2 - res
        if c1 < 0:
            c1 = 0
            c2 = res
        elif c2 > cm:
            c1 = cm-res
            c2 = cm
    data_npys = [data_npy[a1:a2,b1:b2,c1:c2] for data_npy in data_npys]
    label_npys = [label_npy[a1:a2,b1:b2,c1:c2] for label_npy in label_npys]
    for label_npy in label_npys:
        label_npy[label_npy <= 0] = 0
        label_npy[label_npy > 0] = 1
    data_npys = [np.pad(data_npy,((pada//2,pada-pada//2),(padb//2,padb-padb//2),(padc//2,padc-padc//2))) for data_npy in data_npys]
    label_npys = [np.pad(label_npy,((pada//2,pada-pada//2),(padb//2,padb-padb//2),(padc//2,padc-padc//2))) for label_npy in label_npys]
    
    return data_npys, label_npys


filess = [[],[],[],[],[],[],[],[],[],[]]
for file in os.listdir('/root/data/segmentation'):
    filess[int(file.split('_')[0])-1].append(file)
filess = [sorted(files, key=cmp_to_key(cmpfun)) for files in filess]

train_files = []
test_files = []
for files in filess:
    images = ['/root/data/source/'+file for file in files]
    labels = ['/root/data/segmentation/'+file for file in files]
    for i in range(2,5):
        test_files.append({'image': images[i-2:i+1], 'label': labels[i-2:i+1]})
    for i in range(5,len(labels)):
        train_files.append({'image': images[i-2:i+1], 'label': labels[i-2:i+1]})


