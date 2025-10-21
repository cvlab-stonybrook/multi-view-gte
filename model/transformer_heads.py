import torch
from torch import nn


class Prediction_Head(nn.Module):
    def __init__(self, in_channels):
        
        super(Prediction_Head, self).__init__()
         # encoding for saliency
        self.compress_conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0)
        self.compress_bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        
        self.resize_func = nn.Upsample(size=(32,32), mode='bilinear', align_corners=False)
        # decoding
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 64x64
        self.deconv_bn1 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(128, 64, 3, padding=3, dilation=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=3, dilation=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1) 
        
    def forward(self, x):
        
        x = self.compress_conv1(x)
        x = self.compress_bn1(x)
        x = self.relu(x)
        x = self.resize_func(x)
        
        x = self.relu(self.deconv_bn1(self.deconv1(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x
        
        
# predicts in vs out gaze; CompressModality + Linear
class Inout_Head(nn.Module):
    
    def __init__(self, in_channels):
        
        '''
        args:
        in_channels: number of input channels
        '''
        self.relu = nn.ReLU(inplace=True)
        self.compress_conv1_inout = nn.Conv2d(in_channels, 512, kernel_size=3, stride=2, dilation=2, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.inout = nn.Sequential(nn.Linear(1024, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        
        
    def forward(self, x, head_embedding):
        
        x = self.compress_conv1_inout(x)
        x = self.compress_bn1_inout(x)
        x = self.relu(x)
        x = self.compress_conv2_inout(x)
        x = self.compress_bn2_inout(x)
        x = self.relu(x)
        x = self.maxpool(x.shape[2])(x).squeeze(dim=-1).squeeze(dim=-1)
        
        x = torch.cat([x, head_embedding], axis=1)
        x = self.inout(x)        
        return x
    
    
    