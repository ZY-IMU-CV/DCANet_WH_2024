import torch
import torch.nn as nn
from torch.nn import functional as F
def activation():
    return nn.ReLU(inplace=True)
def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.act=activation()
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=True)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=(1,1),groups=out_channels, padding=dilations,dilation=dilations,bias=True)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=True)
        self.act3=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        x=self.conv1(x)
        x = self.act1(x)
        if self.stride == 2:
            x=self.avg(x)
        x=self.conv2(x)
        x=self.act2(x)
        x=self.conv3(x)
        x = self.act3(x)
        return x

class D11(nn.Module):
    def __init__(self):
        super(D11, self).__init__()
        self.stage4 = DBlock(32, 48, [1], 2)
        self.stage8 = nn.Sequential(
            DBlock(48, 128, [1], 2),
            DBlock(128, 128, [1],1),
            DBlock(128, 128, [1], 1)
        )
        self.stage16 = nn.Sequential(
            DBlock(128, 256, [1], 2),
            DBlock(256, 256, [2], 1),
            DBlock(256, 256, [2], 1),
            DBlock(256, 256, [4], 1),
            DBlock(256, 256, [4], 1),
            DBlock(256, 256, [8], 1),
            DBlock(256, 256, [8], 1)
        )
    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        return {"4": x4, "8": x8, "16": x16}

class Exp2_Decoder26(nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16=x["4"], x["8"],x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class CLReg(nn.Module):
    def __init__(self):
        super(CLReg,self).__init__()
        self.stem=ConvBnAct(3,32,3,2,1)
        self.body=D11()
        self.decoder = Exp2_Decoder26(19, {"4":48,"8":128,"16":256})
    def forward(self,x):
        x = self.stem(x)
        x = self.body(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(1024,2048), mode='bilinear', align_corners=False)
        return x



