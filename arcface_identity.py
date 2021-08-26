# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:13:34 2021

@author: LanGuipeng
"""
import torch
import torch.nn.functional as F
from torchvision.models import resnet101
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                      
class GETfaceidentity(nn.Module):
    def __init__(self):
        super(GETfaceidentity, self).__init__()

        self.Z = resnet101(num_classes=256)
        self.Z.load_state_dict(torch.load('model/Arcface.pth', map_location='cpu'))


    def forward(self, source_img):
        z_id = self.Z(F.interpolate(source_img, size=112, mode='bilinear', align_corners=False))
        z_id = F.normalize(z_id)
        z_id = z_id.detach()

        return z_id