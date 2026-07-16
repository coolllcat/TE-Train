# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 21:24:34 2022

@author: ASUS
"""
from turtle import forward
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from RATnet import RATNet

class Selfattention(nn.Module):
    """Self-attention with conv_qkv and reattention function"""
    
    def __init__(self, dim_in, dim_out, num_heads, patch_size=2, attn_drop=0., proj_drop=0., init_values=1e-4,
                 kernel_size=3, stride=1, padding=1, conv_qkv=True, layerscale=True,
                 reattention_heads=8, reattention_kernel_size=3, reattention=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = dim_out ** -0.5
        self.conv_qkv = conv_qkv
        self.reattention = reattention
        self.layerscale = layerscale
        
        if self.conv_qkv:
            self.proj_q = nn.Sequential(
                nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in),
                nn.BatchNorm3d(dim_out),
                nn.ReLU(inplace=True))
            self.proj_k = nn.Sequential(
                nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in),
                nn.BatchNorm3d(dim_out),
                nn.ReLU(inplace=True))
            self.proj_v = nn.Sequential(
                nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in),
                nn.BatchNorm3d(dim_out),
                nn.ReLU(inplace=True))
        else:
            self.proj_q = nn.Linear(dim_in*patch_size**3//num_heads, dim_out*patch_size**3//num_heads, bias=False)
            self.proj_k = nn.Linear(dim_in*patch_size**3//num_heads, dim_out*patch_size**3//num_heads, bias=False)
            self.proj_v = nn.Linear(dim_in*patch_size**3//num_heads, dim_out*patch_size**3//num_heads, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if self.reattention:
            self.reattention_expansion = nn.Linear(num_heads,reattention_heads,bias=False)
            self.reattention_conv = nn.Conv2d(reattention_heads, reattention_heads, 
                                              kernel_size=reattention_kernel_size, 
                                              padding=reattention_kernel_size//2,stride=1,
                                              bias=False,groups=reattention_heads) # depthwise-conv
            self.reattention_reduction = nn.Linear(reattention_heads,num_heads,bias=False)
        
        if self.layerscale:
            self.LayerScale = nn.Parameter(init_values*torch.ones((dim_out)),requires_grad=True)

    def Patch(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, self.num_heads, C//self.num_heads, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size, D//self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 5, 7, 4, 6, 8, 2) # B*h*H'*W'*D'*k*k*k*d
        x = x.reshape(B, self.num_heads, H*W*D//(self.patch_size**3), self.patch_size**3*C//self.num_heads)
        return x
    
    def unPatch(self, x, B, C, H, W, D):
        x = x.reshape(B, self.num_heads, H//self.patch_size, W//self.patch_size, D//self.patch_size, self.patch_size, self.patch_size, self.patch_size, C//self.num_heads)
        x = x.permute(0, 1, 8, 2, 5, 3, 6, 4, 7) # B*h*d*H'*k*W'*k*D'*k
        x = x.reshape(B, C, H, W, D)
        return x
    
    def forward(self, x_q, x_k, x_v):
        B1, C1, H1, W1, D1 = x_q.shape
        B2, C2, H2, W2, D2 = x_k.shape
        
        if self.conv_qkv:# B*C*H*W*D
            q = self.proj_q(x_q)
            k = self.proj_k(x_k)
            v = self.proj_v(x_v) # B*C*H*W*D
            q = self.Patch(q) # B*H*N1*d
            k = self.Patch(k)
            v = self.Patch(v) # B*H*N2*d
        else:
            q = self.Patch(x_q) # B*H*N1*d
            k = self.Patch(x_k)
            v = self.Patch(x_v) # B*H*N2*d
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)
            # B*C*H*W*D

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale # B*H*N1*N2
        attn = F.softmax(attn_score, dim=-1)
        if self.reattention:
            attn = self.reattention_expansion(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B*H*N1*N2
            attn = self.reattention_conv(attn) # B*H*N1*N2
            attn = self.reattention_reduction(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # B*H*N1*N2  
        attn = self.attn_drop(attn)
        v_ = torch.einsum('bhlt,bhtv->bhlv', [attn, v]) # B*H*N1*d

        attn_score2 = torch.einsum('bhlk,bhtk->bhlt', [k, v_]) * self.scale # B*H*N1*N1
        attn2 = F.softmax(attn_score2, dim=-1)
        attn2 = self.attn_drop(attn2)
        x = torch.einsum('bhlt,bhtv->bhlv', [attn2, v_]) # B*H*N1*d
        
        x = self.unPatch(x, B2, self.dim_out, H2, W2, D2) # B1*C1*H1*W1*D1
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.layerscale:
            x = x * self.LayerScale
        x = x.permute(0, 4, 1, 2, 3)
             
        return x
    
class FFN(nn.Module):
    def __init__(self, d_in, d_hid, d_out=None, dropout=0.1):
        super().__init__()
        if d_out is None:
            d_out = d_hid
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_out)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        
        x = x.permute(0,4,1,2,3)
        
        return x
    
class Norm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
    
class CircleTransFormer(nn.Module):
    def __init__(self, dim_in, dim_out, heads=4, patch_size=4,
                 conv_qkv=True, reattention=True, layerscale=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.SA = Selfattention(dim_in, dim_in, heads, patch_size=patch_size, 
                                conv_qkv=conv_qkv, reattention=reattention, layerscale=layerscale)
        self.FFN = FFN(dim_in, dim_out)
        self.Norm1 = Norm(dim_in)
        self.Norm2 = Norm(dim_out)
        if dim_out != dim_in:
            self.downconv = nn.Sequential(
                            nn.Conv3d(dim_in, dim_out, 1),
                            nn.BatchNorm3d(dim_out, momentum=bn_mom),
                            nn.ReLU(inplace=True))
        
    def forward(self, x_static, x_dynamic):
        residual = x_static
        x_static = self.SA(x_dynamic, x_static, x_static)
        x_static = self.Norm1(x_static)
        x_static += residual
        
        residual = self.downconv(x_static) if self.dim_out != self.dim_in else x_static
        x_static = self.FFN(x_static)
        x_static = self.Norm2(x_static)
        x_static += residual
        
        return x_static

class Video3D(nn.Module):
    def __init__(self, basemodel, dim, heads, patch_size):
        super().__init__()
        self.model = basemodel
        self.projt = CircleTransFormer(dim, dim, heads=heads, patch_size=patch_size)

    def forward(self, xlist):
        bs = xlist[0].shape[0]
        fea_in = self.model.encode(torch.cat(xlist, dim=0))
        fea_in = torch.split(fea_in, bs, dim=0)

        static1 = fea_in[-1]
        dynamic1 = fea_in[:-1]
        circle_out1 = self.projt(static1, torch.cat(dynamic1, dim=2))
        static2 = fea_in[-2]
        dynamic2 = fea_in[::2]
        circle_out2 = self.projt(static2, torch.cat(dynamic2, dim=2))
        static3 = fea_in[0]
        dynamic3 = fea_in[1:]
        circle_out3 = self.projt(static3, torch.cat(dynamic3, dim=2))

        fea_out = self.model.decode(torch.cat([*fea_in, circle_out3, circle_out2, circle_out1], dim=0))
        out = torch.split(fea_out, bs, dim=0)

        return out


if __name__ == "__main__":
    data = torch.rand(1,1,96,96,96)
    model = Video3D(RATNet('basic', [2, 2, 2, 2]), 32, 4, 4)
    print(model([data,data,data]).shape)


    