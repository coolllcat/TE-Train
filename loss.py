# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:39:49 2022

@author: ASUS
"""
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
from math import exp,log
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
class Segloss(nn.Module):
    def __init__(self, n_classes, weight=torch.tensor([0.3,0.7])):
        super(Segloss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.to(device)

    def One_Hot(self, tensor, num_classes):
        B, _, H, W, D = tensor.shape
        one_hot = torch.zeros(B, num_classes, H, W, D).long().to(device)
        one_hot.scatter_(dim=1, index=tensor, src=torch.ones(B, num_classes, H, W, D).long().to(device))
        return one_hot

    def forward(self, inputs, target):
        smooth = 0.01
        b, c, h, w, d = inputs.size()

        inputs1 = F.softmax(inputs, dim=1).view(b, self.n_classes, -1)
        target1 = self.One_Hot(target, self.n_classes).view(b, self.n_classes, -1)
        inputs1 = inputs1[:,1:,:]
        target1 = target1[:,1:,:]
        inter = torch.sum(inputs1 * target1, 2) + smooth
        union = torch.sum(inputs1, 2) + torch.sum(target1, 2) + smooth
        score = 1 - torch.mean(2.0 * inter / union)
        
        loss = F.cross_entropy(inputs.permute(0,2,3,4,1).reshape(-1,c), target.view(-1), self.weight)

        total_loss = score + loss

        return total_loss

        
def Dnetloss(logits_s, logits_t, mode):
    if mode == 'D':
        d1 = F.cross_entropy(logits_s, torch.ones(logits_s.shape[0],dtype=torch.long).to(device))
        d2 = F.cross_entropy(logits_t, torch.zeros(logits_t.shape[0],dtype=torch.long).to(device))
        return d1 + d2
    elif mode == 'G':
        d1 = F.cross_entropy(logits_s, torch.ones(logits_s.shape[0],dtype=torch.long).to(device))
        d2 = F.cross_entropy(logits_t, torch.ones(logits_t.shape[0],dtype=torch.long).to(device))
        return d1 + d2
    else:
        return None


def IWloss(F):
    B, C, H, W, D = F.shape
    F1, F2 = torch.split(F, C//2, dim=1)
    F1 = F1.reshape(B, C//2, -1)
    F2 = F2.reshape(B, C//2, -1)
    F1 = F1 - torch.mean(F1, dim=2, keepdim=True)
    F2 = F2 - torch.mean(F2, dim=2, keepdim=True)
    Cov11 = torch.mean(F1 * F1, dim=2)
    Cov12 = torch.mean(F1 * F2, dim=2)
    Cov21 = torch.mean(F2 * F1, dim=2)
    Cov22 = torch.mean(F2 * F2, dim=2)
    L = torch.abs(Cov11-1) + torch.abs(Cov22-1) + torch.abs(Cov12) + torch.abs(Cov21)
    return L.mean()

