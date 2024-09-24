import torch
import torch.nn as nn
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
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x


class D11(nn.Module):
    def __init__(self):
        super(D11, self).__init__()
        self.stage4_conv1 = nn.Conv2d(32, 48, kernel_size=(1, 1), bias=True)

        self.stage4_conv2 = nn.Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), groups=48, padding=1, dilation=(1, 1),
                                      bias=True)
        self.stage4_conv3 = nn.Conv2d(48, 48, kernel_size=(1, 1), bias=True)

        self.stage8_0_conv1 = nn.Conv2d(48, 128, kernel_size=(1, 1), bias=True)
        self.stage8_0_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), groups=128, padding=1,
                                        dilation=(1, 1), bias=True)
        self.stage8_0_conv3 = nn.Conv2d(128, 128, kernel_size=(1, 1), bias=True)

        self.stage8_1_conv1 = nn.Conv2d(128, 128, kernel_size=(1, 1), bias=True)
        self.stage8_1_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), groups=128, padding=1,
                                        dilation=(1, 1), bias=True)
        self.stage8_1_conv3 = nn.Conv2d(128, 128, kernel_size=(1, 1), bias=True)

        self.stage8_2_conv1 = nn.Conv2d(128, 128, kernel_size=(1, 1), bias=True)
        self.stage8_2_conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), groups=128, padding=1,
                                        dilation=(1, 1), bias=True)
        self.stage8_2_conv3 = nn.Conv2d(128, 128, kernel_size=(1, 1), bias=True)

        self.stage16_0_conv1 = nn.Conv2d(128, 256, kernel_size=(1, 1), bias=True)
        self.stage16_0_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), groups=256, padding=1,
                                         dilation=(1, 1), bias=True)
        self.stage16_0_conv3 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)

        self.stage16_1_conv1 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)
        self.stage16_1_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), groups=256, padding=2,
                                         dilation=(2, 2), bias=True)
        self.stage16_1_conv3 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)

        self.stage16_2_conv1 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)
        self.stage16_2_conv2= nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), groups=256, padding=2,
                                          dilation=2, bias=True)

        self.stage16_2_conv3 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)

        self.stage16_3_conv1 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)

        self.stage16_3_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), groups=256, padding=4,
                                          dilation=4, bias=True)

        self.stage16_3_conv3 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)

        self.stage16_4_conv1 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)
        self.stage16_4_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), groups=256, padding=4,
                                         dilation=4, bias=True)
        self.stage16_4_conv3 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)

        self.stage16_5_conv1 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)
        self.stage16_5_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), groups=256, padding=8,
                                         dilation=8, bias=True)
        self.stage16_5_conv3 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)

        self.stage16_6_conv1 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)
        self.stage16_6_conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), groups=256, padding=8,
                                         dilation=8, bias=True)
        self.stage16_6_conv3 = nn.Conv2d(256, 256, kernel_size=(1, 1), bias=True)
        self.downsample = nn.AvgPool2d(2,2,ceil_mode=True)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x4 = self.stage4_conv1(x)
        x4 = self.act(x4)
        x4 = self.downsample(x4)#avgpool
        x4 = self.stage4_conv2(x4)
        x4 = self.act(x4)
        x4 = self.stage4_conv3(x4)
        x4 = self.act(x4)

        x80 = self.stage8_0_conv1(x4)
        x80 = self.act(x80)
        x80 = self.downsample(x80)
        x80 = self.stage8_0_conv2(x80)
        x80 = self.act(x80)
        x80 = self.stage8_0_conv3(x80)
        x80 = self.act(x80)

        x81 = self.stage8_1_conv1(x80)
        x81 = self.act(x81)
        x81 = self.stage8_1_conv2(x81)
        x81 = self.act(x81)
        x81 = self.stage8_1_conv3(x81)
        x81 = self.act(x81)

        x8 = self.stage8_2_conv1(x81)
        x8 = self.act(x8)
        x8 = self.stage8_2_conv2(x8)
        x8 = self.act(x8)
        x8 = self.stage8_2_conv3(x8)
        x8 = self.act(x8)

        x160 = self.stage16_0_conv1(x8)
        x160 = self.act(x160)
        x160 = self.downsample(x160)
        x160 = self.stage16_0_conv2(x160)
        x160 = self.act(x160)
        x160 = self.stage16_0_conv3(x160)
        x160 = self.act(x160)

        x161 = self.stage16_1_conv1(x160)
        x161 = self.act(x161)
        x161 = self.stage16_1_conv2(x161)
        x161 = self.act(x161)
        x161 = self.stage16_1_conv3(x161)
        x161 = self.act(x161)

        x162 = self.stage16_2_conv1(x161)
        x162 = self.act(x162)
        x162 = self.stage16_2_conv2(x162)
        x162 = self.act(x162)
        x162 = self.stage16_2_conv3(x162)
        x162 = self.act(x162)

        x163 = self.stage16_3_conv1(x162)
        x163 = self.act(x163)
        x163 = self.stage16_3_conv2(x163)
        x163 = self.act(x163)
        x163 = self.stage16_3_conv3(x163)
        x163 = self.act(x163)

        x164 = self.stage16_4_conv1(x163)
        x164 = self.act(x164)
        x164 = self.stage16_4_conv2(x164)
        x164 = self.act(x164)
        x164 = self.stage16_4_conv3(x164)
        x164 = self.act(x164)

        x165 = self.stage16_5_conv1(x164)
        x165 = self.act(x165)
        x165 = self.stage16_5_conv2(x165)
        x165 = self.act(x165)
        x165 = self.stage16_5_conv3(x165)
        x165 = self.act(x165)

        x16 = self.stage16_6_conv1(x165)
        x16 = self.act(x16)
        x16 = self.stage16_6_conv2(x16)
        x16 = self.act(x16)
        x16 = self.stage16_6_conv3(x16)
        x16 = self.act(x16)
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
        self.decoder = Exp2_Decoder26(19,  {"4":48,"8":128,"16":256})

    def forward(self,x):
        x = self.stem(x)
        x = self.body(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=(1024,2048), mode='bilinear', align_corners=False)
        return x

