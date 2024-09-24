from torch import nn
import torch
from torch.nn import functional as F

def activation():
    return nn.ReLU(inplace=True)
def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x,dataset):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=(1,1),groups=out_channels, padding=dilations,dilation=dilations,bias=False)
        self.bn2 = norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3 = norm2d(out_channels)
        self.act3=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
        #self.short = Shortcut(out_channels,groups)
    def forward(self, x,dataset):
        x=self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.stride == 2:
            x=self.avg(x)
        x=self.conv2(x)+x
        x = self.bn2(x)
        x=self.act2(x)
        x=self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x


class eleven_Decoder0(nn.Module):#Sum+Cat
    def __init__(self, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier_city=nn.Conv2d(64, 19, (1,1))
        self.classifier_bdd = nn.Conv2d(64, 18, (1, 1))
        
    def forward(self, x,dataset):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32,dataset)
        x16=self.head16(x16,dataset)
        x8=self.head8(x8,dataset)
        x4=self.head4(x4,dataset)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16,dataset)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8,dataset)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4,dataset)
        if dataset == "Cityscapes":
            x4=self.classifier_city(x4)
        elif dataset == "BDD100K":
            x4 = self.classifier_bdd(x4)
        return x4


def generate_stage2(ds,block_fun):
    blocks=[]
    for d in ds:
        blocks.append(block_fun(d))
    return blocks

class RegSegBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage4=DBlock(32, 48, [1], 2)
        self.stage8_1=DBlock(48, 128,  [1], 2)
        self.stage8_2=DBlock(128, 128, [1], 1)
        self.stage8_3=DBlock(128, 128, [1], 1)

        self.stage16_1=DBlock(128, 256, [1], 2)
        self.stage16_2=DBlock(256, 256, [2], 1)
        self.stage16_3=DBlock(256, 256, [4], 1)
        
        self.stage32_1=DBlock(256, 256, [1], 1)
        self.stage32_2=DBlock(256, 256, [4], 1)
        self.stage32_3=DBlock(256, 256, [8], 1)

    def forward(self,x,dataset):
        x4=self.stage4(x,dataset)
        x8=self.stage8_1(x4,dataset)
        x8 = self.stage8_2(x8, dataset)
        x8 = self.stage8_3(x8, dataset)
        x16 = self.stage16_1(x8 , dataset)
        x16 = self.stage16_2(x16, dataset)
        x16 = self.stage16_3(x16, dataset)
        x32 = self.stage32_1(x16, dataset)
        x32 = self.stage32_2(x32, dataset)
        x32 = self.stage32_3(x32, dataset)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}