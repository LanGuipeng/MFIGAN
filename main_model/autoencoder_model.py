# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:42:56 2020

@author: LanGuipeng
"""
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.ReflectionPad2d = nn.ReflectionPad2d(1)
        self.Conv2d = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d = nn.InstanceNorm2d(num_features = in_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.ReflectionPad2d(x)
        out = self.Conv2d(out)
        out = self.InstanceNorm2d(out)
        out = self.relu(out)
        out = self.ReflectionPad2d(out)
        out = self.Conv2d(out)
        out = self.InstanceNorm2d(out)
        ResidualBlock_out = x  + out

        return ResidualBlock_out

class Encoder(nn.Module):
    def __init__(self, input_nc, n_residual_blocks=4):
        super(Encoder, self).__init__()

        self.ReflectionPad2d_init_conv_block = nn.ReflectionPad2d(3)
        self.Conv2d_init_conv_block = nn.Conv2d(in_channels=input_nc, out_channels=32, kernel_size=7, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_init_conv_block = nn.InstanceNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        # Downsampling Residual_blocks1: H*H*32--->(H/2)*(H/2)*64
        in_features = 32
        out_features = in_features*2
        self.ReflectionPad2d_DR_1 = nn.ReflectionPad2d(1)
        self.Conv2d_DR_1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_DR_1 = nn.InstanceNorm2d(out_features)
        self.AvgPool2d_DR_1 = nn.AvgPool2d(2)
        # number of Residual blocks: 4
        in_features = out_features
        self.Residualblocks_DR_1_1 = ResidualBlock(in_features)
        self.Residualblocks_DR_1_2 = ResidualBlock(in_features)
        # self.Residualblocks_DR_1_3 = ResidualBlock(in_features)
        # self.Residualblocks_DR_1_4 = ResidualBlock(in_features)
        
        # Downsampling Residual_blocks2: (H/2)*(H/2)*64--->(H/4)*(H/4)*128
        in_features = 64
        out_features = in_features*2
        self.ReflectionPad2d_DR_2 = nn.ReflectionPad2d(1)
        self.Conv2d_DR_2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_DR_2 = nn.InstanceNorm2d(out_features)
        self.AvgPool2d_DR_2 = nn.AvgPool2d(2)
        # number of Residual blocks: 4
        in_features = out_features
        self.Residualblocks_DR_2_1 = ResidualBlock(in_features)
        self.Residualblocks_DR_2_2 = ResidualBlock(in_features)
        # self.Residualblocks_DR_2_3 = ResidualBlock(in_features)
        # self.Residualblocks_DR_2_4 = ResidualBlock(in_features)
        
        # Downsampling Residual_blocks3: (H/4)*(H/4)*128--->(H/8)*(H/8)*256
        in_features = 128
        out_features = in_features*2
        self.ReflectionPad2d_DR_3 = nn.ReflectionPad2d(1)
        self.Conv2d_DR_3 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_DR_3 = nn.InstanceNorm2d(out_features)
        self.AvgPool2d_DR_3 = nn.AvgPool2d(2)
        # number of Residual blocks: 4
        in_features = out_features
        self.Residualblocks_DR_3_1 = ResidualBlock(in_features)
        self.Residualblocks_DR_3_2 = ResidualBlock(in_features)
        # self.Residualblocks_DR_3_3 = ResidualBlock(in_features)
        # self.Residualblocks_DR_3_4 = ResidualBlock(in_features)
        
        # Downsampling Residual_blocks4: (H/8)*(H/8)*256--->(H/16)*(H/16)*512
        in_features = 256
        out_features = in_features*2
        self.ReflectionPad2d_DR_4 = nn.ReflectionPad2d(1)
        self.Conv2d_DR_4 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_DR_4 = nn.InstanceNorm2d(out_features)
        self.AvgPool2d_DR_4 = nn.AvgPool2d(2)
        # number of Residual blocks: 4
        in_features = out_features
        self.Residualblocks_DR_4_1 = ResidualBlock(in_features)
        self.Residualblocks_DR_4_2 = ResidualBlock(in_features)
        # self.Residualblocks_DR_4_3 = ResidualBlock(in_features)
        # self.Residualblocks_DR_4_4 = ResidualBlock(in_features)
        
        
        # feature map compress
        self.ReflectionPad2d_feature_map_compress = nn.ReflectionPad2d(1)
        self.Conv2d_feature_map_compress = nn.Conv2d(in_channels=512, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_feature_map_compress = nn.InstanceNorm2d(32)
        
    def forward(self, x):
        # Initial convolution block 
        out =  self.ReflectionPad2d_init_conv_block(x)
        out = self.Conv2d_init_conv_block(out)
        out = self.InstanceNorm2d_init_conv_block(out)
        out = self.relu(out)
        # Downsampling Residual_blocks1
        out = self.ReflectionPad2d_DR_1(out)
        out = self.Conv2d_DR_1(out)
        out = self.InstanceNorm2d_DR_1(out)
        out = self.relu(out)
        out = self.AvgPool2d_DR_1(out)
        out = self.Residualblocks_DR_1_1(out)
        out = self.Residualblocks_DR_1_2(out)
        # out = self.Residualblocks_DR_1_3(out)
        # out = self.Residualblocks_DR_1_4(out)
        # Downsampling Residual_blocks2
        out = self.ReflectionPad2d_DR_2(out)
        out = self.Conv2d_DR_2(out)
        out = self.InstanceNorm2d_DR_2(out)
        out = self.relu(out)
        out = self.AvgPool2d_DR_2(out)
        out = self.Residualblocks_DR_2_1(out)
        out = self.Residualblocks_DR_2_2(out)
        # out = self.Residualblocks_DR_2_3(out)
        # out = self.Residualblocks_DR_2_4(out)
        # Downsampling Residual_blocks3
        out = self.ReflectionPad2d_DR_3(out)
        out = self.Conv2d_DR_3(out)
        out = self.InstanceNorm2d_DR_3(out)
        out = self.relu(out)
        out = self.AvgPool2d_DR_3(out)
        out = self.Residualblocks_DR_3_1(out)
        out = self.Residualblocks_DR_3_2(out)
        # out = self.Residualblocks_DR_3_3(out)
        # out = self.Residualblocks_DR_3_4(out)
        # Downsampling Residual_blocks4
        out = self.ReflectionPad2d_DR_4(out)
        out = self.Conv2d_DR_4(out)
        out = self.InstanceNorm2d_DR_4(out)
        out = self.relu(out)
        out = self.AvgPool2d_DR_4(out)
        out = self.Residualblocks_DR_4_1(out)
        out = self.Residualblocks_DR_4_2(out)
        # out = self.Residualblocks_DR_4_3(out)
        # out = self.Residualblocks_DR_4_4(out)
        # feature map compress
        out = self.ReflectionPad2d_feature_map_compress(out)
        out = self.Conv2d_feature_map_compress(out)
        out = self.InstanceNorm2d_feature_map_compress(out)
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, output_nc, n_residual_blocks=4):
        super(Decoder, self).__init__()
        # input_nc = 16 single image
        # initial residual blocks:number of channels: 16--->128--->256--->384--->512 no change the size of feature map
        self.ReflectionPad2d_init_residual_block = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        
        in_features = 16
        out_features = 128
        self.Conv2d_init_residual_block_1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_init_residual_block_1 = nn.InstanceNorm2d(out_features)
        in_features = out_features
        self.Residualblocks_DR_DE_1_1 = ResidualBlock(in_features)
        # self.Residualblocks_DR_DE_1_2 = ResidualBlock(in_features)
        
        in_features = 128
        out_features = 256
        self.Conv2d_init_residual_block_2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_init_residual_block_2 = nn.InstanceNorm2d(out_features)
        in_features = out_features
        self.Residualblocks_DR_DE_2_1 = ResidualBlock(in_features)
        # self.Residualblocks_DR_DE_2_2 = ResidualBlock(in_features)
        
        in_features = 256
        out_features = 384
        self.Conv2d_init_residual_block_3 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_init_residual_block_3 = nn.InstanceNorm2d(out_features)
        in_features = out_features
        self.Residualblocks_DR_DE_3_1 = ResidualBlock(in_features)
        # self.Residualblocks_DR_DE_3_2 = ResidualBlock(in_features)
        
        in_features = 384
        out_features = 512
        self.Conv2d_init_residual_block_4 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=0, bias=True)
        self.InstanceNorm2d_init_residual_block_4 = nn.InstanceNorm2d(out_features)
        in_features = out_features
        self.Residualblocks_DR_DE_4_1 = ResidualBlock(in_features)
        # self.Residualblocks_DR_DE_4_2 = ResidualBlock(in_features)
        
        # upsampling:feature map: 64*64--->128*128--->256*256--->512*512
        in_features = 512
        out_features = 512
        self.ConvTranspose2d_up_1 = nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1)
        self.InstanceNorm2d_up_1 = nn.InstanceNorm2d(out_features)
        in_features = out_features
        self.Residualblocks_up_1_1 = ResidualBlock(in_features)
        # self.Residualblocks_up_1_2 = ResidualBlock(in_features)
        
        in_features = 512
        out_features = 512
        self.ConvTranspose2d_up_2 = nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1)
        self.InstanceNorm2d_up_2 = nn.InstanceNorm2d(out_features)
        in_features = out_features
        self.Residualblocks_up_2_1 = ResidualBlock(in_features)
        # self.Residualblocks_up_2_2 = ResidualBlock(in_features)
        
        in_features = 512
        out_features = 256
        self.ConvTranspose2d_up_3 = nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1)
        self.InstanceNorm2d_up_3 = nn.InstanceNorm2d(out_features)
        in_features = out_features
        self.Residualblocks_up_3_1 = ResidualBlock(in_features)
        # self.Residualblocks_up_3_2 = ResidualBlock(in_features)
            
        in_features = 256
        out_features = 128
        self.ConvTranspose2d_up_4 = nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1)
        self.InstanceNorm2d_up_4 = nn.InstanceNorm2d(out_features)
        in_features = out_features
        self.Residualblocks_up_4_1 = ResidualBlock(in_features)
        # self.Residualblocks_up_4_2 = ResidualBlock(in_features)

        # Output layer
        self.ReflectionPad2d_out_layer = nn.ReflectionPad2d(3)
        self.Conv2d_out_layer = nn.Conv2d(in_channels=128, out_channels=output_nc, kernel_size=7, stride=1, padding=0, bias=True)
        self.tanh_out_layer = nn.Tanh()
        
    def forward(self, x):
        # initial residual blocks
        out = self.ReflectionPad2d_init_residual_block(x)
        out = self.Conv2d_init_residual_block_1(out)
        out = self.InstanceNorm2d_init_residual_block_1(out)
        out = self.relu(out)
        out = self.Residualblocks_DR_DE_1_1(out)
        # out = self.Residualblocks_DR_DE_1_2(out)
        
        out = self.ReflectionPad2d_init_residual_block(out)
        out = self.Conv2d_init_residual_block_2(out)
        out = self.InstanceNorm2d_init_residual_block_2(out)
        out = self.relu(out)
        out = self.Residualblocks_DR_DE_2_1(out)
        # out = self.Residualblocks_DR_DE_2_2(out)
        
        out = self.ReflectionPad2d_init_residual_block(out)
        out = self.Conv2d_init_residual_block_3(out)
        out = self.InstanceNorm2d_init_residual_block_3(out)
        out = self.relu(out)
        out = self.Residualblocks_DR_DE_3_1(out)
        # out = self.Residualblocks_DR_DE_3_2(out)
        
        out = self.ReflectionPad2d_init_residual_block(out)
        out = self.Conv2d_init_residual_block_4(out)
        out = self.InstanceNorm2d_init_residual_block_4(out)
        out = self.relu(out)
        # out = self.Residualblocks_DR_DE_4_1(out)
        # out = self.Residualblocks_DR_DE_4_2(out)
        
        # upsampling:
        out = self.ConvTranspose2d_up_1(out)
        out = self.InstanceNorm2d_up_1(out)
        out = self.relu(out)
        out = self.Residualblocks_up_1_1(out)
        # out = self.Residualblocks_up_1_2(out)
        
        out = self.ConvTranspose2d_up_2(out)
        # out = self.InstanceNorm2d_up_2(out)
        out = self.relu(out)
        # out = self.Residualblocks_up_2_1(out)
        # out = self.Residualblocks_up_2_2(out)

        out = self.ConvTranspose2d_up_3(out)
        out = self.InstanceNorm2d_up_3(out)
        out = self.relu(out)
        out = self.Residualblocks_up_3_1(out)
        # out = self.Residualblocks_up_3_2(out)

        out = self.ConvTranspose2d_up_4(out)
        out = self.InstanceNorm2d_up_4(out)
        out = self.relu(out)
        # out = self.Residualblocks_up_4_1(out)
        # out = self.Residualblocks_up_4_2(out)
        
        # Output layer
        out = self.ReflectionPad2d_out_layer(out)
        out = self.Conv2d_out_layer(out)
        out = self.tanh_out_layer(out)
        
        return out

# Discriminator: copy cycleGAN    
# class Discriminator(nn.Module):
#     def __init__(self, input_nc):
#         super(Discriminator, self).__init__()

#         # A bunch of convolutions one after another
#         self.dis_conv_1 = nn.Conv2d(input_nc, 64, 4, stride=2, padding=1)
#         self.dis_leakyrule = nn.LeakyReLU(0.2, inplace=True)
#         self.dis_conv_2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
#         self.dis_InstanceNorm2d_2 = nn.InstanceNorm2d(128)
#         self.dis_conv_3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
#         self.dis_InstanceNorm2d_3 = nn.InstanceNorm2d(256)
#         self.dis_conv_4 = nn.Conv2d(256, 512, 4, padding=1)
#         self.dis_InstanceNorm2d_4 = nn.InstanceNorm2d(512)
#          # FCN classification layer
#         self.dis_FCN_classification_layer = nn.Conv2d(512, 1, 4, padding=1)
       
#     def forward(self, x):
#         out = self.dis_conv_1(x)
#         out = self.dis_leakyrule(out)
#         out = self.dis_conv_2(out)
#         out = self.dis_InstanceNorm2d_2(out)
#         out = self.dis_leakyrule(out)
#         out = self.dis_conv_3(out)
#         out = self.dis_InstanceNorm2d_3(out)
#         out = self.dis_leakyrule(out)
#         out = self.dis_conv_4(out)
#         out = self.dis_InstanceNorm2d_4(out)
#         out = self.dis_leakyrule(out)
#         out = self.dis_FCN_classification_layer(out
#                                                 )
#         # Average pooling and flatten
#         out = F.avg_pool2d(out, out.size()[2:]).view(out.size()[0], -1)
#         return out


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
# # Test the structure of the model   
# from torchsummary import summary
# import torch
# input_nc = 3
# output_nc = 3
# device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
# test = Decoder(input_nc).to(device)
# summary(test,(16,16,16))