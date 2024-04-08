# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:57:08 2021

@author: LanGuipeng
在cycleGAN的基础上加入中间层，中间层的目的是为了将身份加入到训练中
"""
import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Adaptiveidentity(nn.Module):
    def __init__(self, dim):
        super(Adaptiveidentity, self).__init__()
        self.mask = nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Sigmoid())
        
    def forward(self, embedding_faceA, embedding_faceB):
        identity_mask = self.mask(embedding_faceA)
        
        x = embedding_faceA*identity_mask + embedding_faceB*(1-identity_mask)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self, input_nc, n_residual_blocks = 9):
        super(Encoder, self).__init__()
        
        # Inital convolution block
        self.model_init = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), 
                                   nn.InstanceNorm2d(64), nn.ReLU(inplace = True))
        
        # Downsampling
        in_features = 64
        out_features = in_features*2
        self.downsampling_block_1 = nn.Sequential(nn.Conv2d(in_features, out_features, 3, stride = 2, padding =1),
                                                  nn.InstanceNorm2d(out_features), nn.ReLU(inplace = True))        
        in_features = out_features
        out_features = in_features*2
        self.downsampling_block_2 = nn.Sequential(nn.Conv2d(in_features, out_features, 3, stride = 2, padding =1),
                                                  nn.InstanceNorm2d(out_features), nn.ReLU(inplace = True))        
        in_features = out_features
        out_features = in_features*2
        self.downsampling_block_3 = nn.Sequential(nn.Conv2d(in_features, out_features, 3, stride = 2, padding =1),
                                                  nn.InstanceNorm2d(out_features), nn.ReLU(inplace = True))
        in_features = out_features
        out_features = in_features*2
        
        # Residual blocks
        model_residual = []
        for _ in range(n_residual_blocks):
            model_residual += [ResidualBlock(in_features)]
        self.model_residual_block = nn.Sequential(*model_residual)
        
    def forward(self, x):
        x = self.model_init(x)
        x = self.downsampling_block_1(x)
        x = self.downsampling_block_2(x)
        x = self.downsampling_block_3(x)
        x = self.model_residual_block(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, output_nc):
        super(Decoder, self).__init__()
        
        # Upsampling
        in_features = 512
        out_features = in_features // 2
        self.upsampling_block_1 = nn.Sequential(nn.ConvTranspose2d(in_features, out_features, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                                                  nn.InstanceNorm2d(out_features), nn.ReLU(inplace = True))        
        in_features = out_features
        out_features = in_features // 2
        self.upsampling_block_2 = nn.Sequential(nn.ConvTranspose2d(in_features, out_features, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                                                  nn.InstanceNorm2d(out_features), nn.ReLU(inplace = True))   
        in_features = out_features
        out_features = in_features // 2
        self.upsampling_block_3 = nn.Sequential(nn.ConvTranspose2d(in_features, out_features, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
                                                  nn.InstanceNorm2d(out_features), nn.ReLU(inplace = True))   
        in_features = out_features
        out_features = in_features // 2
        
        # Output layer
        self.output_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh())
        
    def forward(self, x):
        x = self.upsampling_block_1(x)
        x = self.upsampling_block_2(x)
        x = self.upsampling_block_3(x)
        x = self.output_layer(x)
            
        return x

class Middle_Residual_block(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Middle_Residual_block, self).__init__()
        
        self.conv_init = nn.Conv2d(input_nc, output_nc, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.conv1 = nn.Conv2d(input_nc, output_nc, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(output_nc, output_nc, kernel_size = 3, stride = 1, padding = 1)
        self.adain1 = AdaptiveInstanceNorm2d(input_nc)
        self.adain2 = AdaptiveInstanceNorm2d(output_nc)
        self.activate = nn.LeakyReLU(0.2, inplace = True)
        
    def forward(self, x):
        x = self.conv_init(x)
        
        adain1 = self.adain1(x.clone())
        adain1 = self.activate(adain1)
        conv_adain1 = self.conv1(adain1)
        
        adain2 = self.adain2(conv_adain1)
        adain2 = self.activate(adain2)
        conv_adain2 = self.conv2(adain2)
        
        out = x + conv_adain2
        
        return out / math.sqrt(2)

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps = 1e-5):
        super(AdaptiveInstanceNorm2d, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.bias = None
        self.weight = None
    
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W) 
        

class Translator(nn.Module):
    def __init__(self, input_nc, inter_nc):
        # input_nc: from Encoder; inter_nc: 64
        super(Translator, self).__init__()
        
        # Conv init
        self.model_init = nn.Conv2d(input_nc, inter_nc, kernel_size = 1, stride = 1, padding = 0)        
        # Adain layer
        self.middle_residual_block_1 = Middle_Residual_block(inter_nc, inter_nc)
        self.middle_residual_block_2 = Middle_Residual_block(inter_nc, inter_nc)
        self.middle_residual_block_3 = Middle_Residual_block(inter_nc, inter_nc)
        self.middle_residual_block_4 = Middle_Residual_block(inter_nc, inter_nc)
        self.middle_residual_block_5 = Middle_Residual_block(inter_nc, inter_nc)
        self.middle_residual_block_6 = Middle_Residual_block(inter_nc, inter_nc)
        self.middle_residual_block_7 = Middle_Residual_block(inter_nc, inter_nc)
        
        self.model = nn.Sequential(self.model_init, 
                                   self.middle_residual_block_1, self.middle_residual_block_2, 
                                   self.middle_residual_block_3, self.middle_residual_block_4, 
                                   self.middle_residual_block_5, self.middle_residual_block_6, 
                                   self.middle_residual_block_7)
        
        self.style_to_params = nn.Linear(256, self.get_num_adain_params(self.model))
        self.features = nn.Sequential(nn.Conv2d(64, input_nc, kernel_size = 1, stride = 1, padding = 0))
        self.masks = nn.Sequential(nn.Conv2d(64, input_nc, kernel_size = 1, stride = 1, padding = 0), 
                                  nn.Sigmoid())
        
    def forward(self, inter_feature, identity):
        p = self.style_to_params(identity)
        self.assign_adain_params(p, self.model)

        mid = self.model(inter_feature)
        f = self.features(mid)
        m = self.masks(mid) 

        return f * m + inter_feature * (1 - m)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                m.bias = adain_params[:, :m.num_features].contiguous().view(-1, m.num_features, 1)
                m.weight = adain_params[:, m.num_features:2 * m.num_features].contiguous().view(-1, m.num_features, 1) + 1
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                num_adain_params += 2 * m.num_features
        return num_adain_params
        
# device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
# test = Translator(512, 64).to(device)
# for name,parameters in test.named_parameters():
#     print(name,':',parameters.size())
