from torch import nn
import torch
from torch import nn as nn
from torch.nn import functional as F


# import SelfAttentionBlock

def activation():
    return nn.ReLU(inplace=True)


def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = norm2d(out_channels)
        if apply_act:
            self.act = activation()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class DBlock_new(nn.Module):
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention="se"):
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        if len(dilations)==1:
            dilation=dilations[0]
            self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        else:
            self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        if attention=="se":
            self.se=SEModule(out_channels,in_channels//4)
        elif attention=="se2":
            self.se=SEModule(out_channels,out_channels//4)
        else:
            self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride,avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        if self.se is not None:
            x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, group_width, stride):
        super().__init__()
        self.stride = stride
        avg_downsample = True
        groups = out_channels // group_width
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm2d(out_channels)
        self.act1 = activation()
        dilation = dilations[0]
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(1, 1), groups=out_channels,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm2d(out_channels)
        self.act2 = activation()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm2d(out_channels)
        self.act3 = activation()
        self.avg = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.stride == 2:
            x = self.avg(x)
        x = self.conv2(x) + x
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x



class DDWblock(nn.Module):  # wh
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DDWblockp(nn.Module):  # wh  更换残差连接位置
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(out_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DDWblock2(nn.Module):  # wh 空洞率设置为6
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = 6

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DDWblock3(nn.Module):  # wh  去掉初始1*1和最终的1*1卷积
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=in_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut2(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # x = self.act3(x + shortcut)
        x = x + shortcut
        return x

class DDWblock4(nn.Module): #wh 将DW中的通道数对比1*1卷积减半
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention="se"): #inchanel=128, outchannel=128
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width  #group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation=dilations

        half_outchannel = out_channels//2

        self.conv1=nn.Conv2d(in_channels, half_outchannel,kernel_size=1,bias=False)#1x1的卷积 128,128
        self.bn1=norm2d(half_outchannel)#BN
        self.act1=activation()#ReLU
        self.conv2=nn.Sequential(
            nn.Conv2d(half_outchannel, half_outchannel,kernel_size=3,stride=stride,padding=dilation,groups=half_outchannel,dilation=dilation,bias=False),
            nn.BatchNorm2d(half_outchannel),
            nn.ReLU(),
            nn.Conv2d(half_outchannel,half_outchannel,kernel_size=1,stride=1,padding=0,bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
            #self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2=norm2d(half_outchannel)#BN
        self.act2=activation()#RELU
        self.conv3=nn.Conv2d(half_outchannel, out_channels,kernel_size=1,bias=False)#1x1卷积
        self.bn3=norm2d(out_channels)#BN
        self.act3=activation()#RELU
        self.se=None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride,avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DDWblock5(nn.Module):  # wh 对stride=1的block的残差连接进行消融实验
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None

        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):

        shortcut = self.shortcut(x) if self.shortcut else torch.zeros_like(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DDWblock6(nn.Module):  # wh 去掉初始1*1卷积
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else torch.zeros_like(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DDWblock7(nn.Module): #wh 将DW中的通道数对比1*1卷积增加一倍
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention="se"): #inchanel=128, outchannel=128
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width  #group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation=dilations

        double_outchannel = out_channels*2

        self.conv1=nn.Conv2d(in_channels, double_outchannel,kernel_size=1,bias=False)#1x1的卷积 128,128
        self.bn1=norm2d(double_outchannel)#BN
        self.act1=activation()#ReLU
        self.conv2=nn.Sequential(
            nn.Conv2d(double_outchannel, double_outchannel,kernel_size=3,stride=stride,padding=dilation,groups=double_outchannel,dilation=dilation,bias=False),
            nn.BatchNorm2d(double_outchannel),
            nn.ReLU(),
            nn.Conv2d(double_outchannel,double_outchannel,kernel_size=1,stride=1,padding=0,bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
            #self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2=norm2d(double_outchannel)#BN
        self.act2=activation()#RELU
        self.conv3=nn.Conv2d(double_outchannel, out_channels,kernel_size=1,bias=False)#1x1卷积
        self.bn3=norm2d(out_channels)#BN
        self.act3=activation()#RELU
        self.se=None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride,avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DPEblock1(nn.Module):
    # DPEblock，stride=2与stride=1时都使用单分支结构
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = None
        else:
            self.shortcut = None

    def forward(self, x):
        #shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x

class DPEblock2(nn.Module):
    # DPEblock，stride=2与stride=1时都使用单分支结构,且不使用第二个1*1卷积
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = None
        else:
            self.shortcut = None

    def forward(self, x):
        #shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = self.act3(x)
        return x

class DPEblock3(nn.Module):  # wh
    #DPE block, 双分支结构的block
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DPEblock4(nn.Module):  # wh
    #DPE block, 双分支结构的block,去掉DW后的1*1卷积
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x + shortcut)
        # if self.se is not None:
        #     x=self.se(x)
        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = self.act3(x + shortcut)
        return x

class DPEblock5(nn.Module):
    # DPEblock，stride=2与stride=1时都使用单分支结构，stride=1时去掉第一个1*1卷积
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=in_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None

        if stride != 1 or in_channels != out_channels:
            self.shortcut = None
        else:
            self.shortcut = None

    def forward(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x

class DPEblock6(nn.Module):  # wh
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DPEblock7(nn.Module):  # wh
    #DPE block, 双分支结构的block,去掉DW后的1*1卷积
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1的卷积 128,128
        self.bn1 = norm2d(out_channels)  # BN
        self.act1 = activation()  # ReLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        #shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        #x = self.conv3(x)
        #x = self.bn3(x)
        #x = self.act3(x + shortcut)
        return x

class DPEblock8(nn.Module):  # wh
    #DPE block, 双分支结构的block,去掉DW后的1*1卷积
    def __init__(self, in_channels, out_channels, dilations, group_width, stride,
                 attention="se"):  # inchanel=128, outchannel=128
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width  # group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation = dilations

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=in_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
        # self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # 1x1卷积
        self.bn3 = norm2d(out_channels)  # BN
        self.act3 = activation()  # RELU
        self.se = None
        # if attention=="se": #SE module
        #     self.se=SEModule(out_channels,in_channels//4)  #outchannel=128 inchannel//4=32
        # elif attention=="se2":
        #     self.se=SEModule(out_channels,out_channels//4)
        # else:
        #     self.se=None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x


class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.act1(self.conv1(y))
        y = self.act2(self.conv2(y))
        return x * y


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg = nn.AvgPool2d(2, 2, ceil_mode=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.avg = None
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.avg is not None:
            x = self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class Shortcut2(nn.Module):  # 去掉avg pooling 后的1*1卷积操作
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut2, self).__init__()
        if avg_downsample and stride != 1:
            self.avg = nn.AvgPool2d(2, 2, ceil_mode=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.avg = None
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.avg is not None:
            x = self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DilatedConv(nn.Module):
    def __init__(self, w, dilations, group_width, stride, bias):
        super().__init__()
        num_splits = len(dilations)
        assert (w % num_splits == 0)
        temp = w // num_splits
        assert (temp % group_width == 0)
        groups = temp // group_width
        convs = []
        for d in dilations:
            convs.append(nn.Conv2d(temp, temp, 3, padding=d, dilation=d, stride=stride, bias=bias, groups=groups))
        self.convs = nn.ModuleList(convs)
        self.num_splits = num_splits

    def forward(self, x):
        x = torch.tensor_split(x, self.num_splits, dim=1)
        res = []
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res, dim=1)


class ConvBnActConv(nn.Module):
    def __init__(self, w, stride, dilation, groups, bias):
        super().__init__()
        self.conv = ConvBnAct(w, w, 3, stride, dilation, dilation, groups)
        self.project = nn.Conv2d(w, w, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.project(x)
        return x


class YBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, group_width, stride):
        super(YBlock, self).__init__()
        groups = out_channels // group_width
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm2d(out_channels)
        self.act1 = activation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=groups,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm2d(out_channels)
        self.act2 = activation()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm2d(out_channels)
        self.act3 = activation()
        self.se = SEModule(out_channels, in_channels // 4)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x


class DilaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, group_width, stride, attention="se"):
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm2d(out_channels)
        self.act1 = activation()
        if len(dilations) == 1:
            dilation = dilations[0]
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=groups,
                                   padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = DilatedConv(out_channels, dilations, group_width=group_width, stride=stride, bias=False)
        self.bn2 = norm2d(out_channels)
        self.act2 = activation()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm2d(out_channels)
        self.act3 = activation()
        if attention == "se":
            self.se = SEModule(out_channels, in_channels // 4)
        elif attention == "se2":
            self.se = SEModule(out_channels, out_channels // 4)
        else:
            self.se = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x


class RegSegBody_17_wh18_add(nn.Module):  # 将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat    之后再进行add
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.conv1 = ConvBnAct(1536, 256, 1)

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            # DBlock(32, 48, [1], gw, 2),
            # DBlock(48,48,[1],gw,1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),

            # DBlock(48, 128, [1], gw, 2),
            # DBlock(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)
        x32_5 = self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_add = x32_2 + x32_4
        x32_add = x32_add + x32_6
        x32_add = x32_add + x32_7
        x32 = x32_add
        # x32_cat = torch.cat((x32_1,x32_3),dim=1)
        # x32_cat = torch.cat((x32_cat, x32_5), dim=1)
        # x32 = torch.cat((x32_cat, x32_7), dim=1)
        # x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        # x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 256}


class RegSegBody_17_wh_dilation_all1(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.conv1 = ConvBnAct(1536, 256, 1)

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )


    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = torch.cat((x32_cat, x32_7), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh_dilation_1234666(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.conv1 = ConvBnAct(1536, 256, 1)

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, 2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, 3, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )


    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = torch.cat((x32_cat, x32_7), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh_dilation_1234468(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.conv1 = ConvBnAct(1536, 256, 1)

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, 2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, 3, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, 8, 1, 1, attention)
        )


    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = torch.cat((x32_cat, x32_7), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num334(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_4}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num345(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_5}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num456(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_6}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num555(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)


        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_5}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num578(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_8(x32_7)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_8}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num1467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            DDWblock(32, 64, dilations1, 1, 2, attention),
            #ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467block1(nn.Module):
    #block结构消融实验，对应表格第一行，即stride=2与stride=1时都使用单分支结构
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock1(64, 128, dilations1, 1, 2, attention),
            DPEblock1(128, 128, dilations1, 1, 1, attention),
            DPEblock1(128, 128, dilations1, 1, 1, attention),
            DPEblock1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock1(128, 256, dilations1, 1, 2, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467block2(nn.Module):
    #block结构消融实验，对应表格第二行，即stride=2与stride=1时都使用单分支结构，且不使用第二个1*1卷积
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock2(64, 128, dilations1, 1, 2, attention),
            DPEblock2(128, 128, dilations1, 1, 1, attention),
            DPEblock2(128, 128, dilations1, 1, 1, attention),
            DPEblock2(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock2(128, 256, dilations1, 1, 2, attention),
            DPEblock2(256, 256, dilations1, 1, 1, attention),
            DPEblock2(256, 256, dilations1, 1, 1, attention),
            DPEblock2(256, 256, dilations1, 1, 1, attention),
            DPEblock2(256, 256, dilations1, 1, 1, attention),
            DPEblock2(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock2(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock2(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock2(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock2(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock2(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock2(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock2(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467block4(nn.Module):
    #block结构消融实验，对应表格第四行，即stride=2时使用双分支结构，stride=1时使用单分支结构，
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock3(64, 128, dilations1, 1, 2, attention),
            DPEblock1(128, 128, dilations1, 1, 1, attention),
            DPEblock1(128, 128, dilations1, 1, 1, attention),
            DPEblock1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock3(128, 256, dilations1, 1, 2, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock3(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467block5(nn.Module):
    #block结构消融实验，对应表格第五行，即stride=2时使用单分支结构，stride=1时使用双分支结构
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock1(64, 128, dilations1, 1, 2, attention),
            DPEblock3(128, 128, dilations1, 1, 1, attention),
            DPEblock3(128, 128, dilations1, 1, 1, attention),
            DPEblock3(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock1(128, 256, dilations1, 1, 2, attention),
            DPEblock3(256, 256, dilations1, 1, 1, attention),
            DPEblock3(256, 256, dilations1, 1, 1, attention),
            DPEblock3(256, 256, dilations1, 1, 1, attention),
            DPEblock3(256, 256, dilations1, 1, 1, attention),
            DPEblock3(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock3(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock3(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock3(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock3(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock3(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock3(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467block6(nn.Module):
    #block结构消融实验，对应表格第六行，即stride=2时使用双分支结构，stride=1时使用双分支结构，同时去掉DW后的1*1卷积
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock4(64, 128, dilations1, 1, 2, attention),
            DPEblock4(128, 128, dilations1, 1, 1, attention),
            DPEblock4(128, 128, dilations1, 1, 1, attention),
            DPEblock4(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock4(128, 256, dilations1, 1, 2, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock4(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock4(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock4(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock4(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock4(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock4(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock4(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467channel5(nn.Module): #修改通道数，对应表格第五行
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":128,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467channel3(nn.Module): #修改通道数，对应表格第五行
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 512, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":512}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467channel2(nn.Module): #修改通道数，对应表格第五行
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 512, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(512, 512, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":512}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467channel1(nn.Module): #修改通道数，对应表格第五行
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(256, 512, dilations1, 1, 2, attention),
            DDWblock(512, 512, dilations1, 1, 1, attention),
            DDWblock(512, 512, dilations1, 1, 1, attention),
            DDWblock(512, 512, dilations1, 1, 1, attention),
            DDWblock(512, 512, dilations1, 1, 1, attention),
            DDWblock(512, 512, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(512, 1024, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(1024, 1024, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(1024, 1024, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(1024, 1024, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(1024, 1024,dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(1024, 1024, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(1024, 1024, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":512}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467block8(nn.Module):
    #block结构消融实验，对应表格第八行，即stride=2与stride=1时都使用单分支结构，同时去掉第一个1*1卷积
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock1(64, 128, dilations1, 1, 2, attention),
            DPEblock5(128, 128, dilations1, 1, 1, attention),
            DPEblock5(128, 128, dilations1, 1, 1, attention),
            DPEblock5(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock1(128, 256, dilations1, 1, 2, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock1(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock5(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock5(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock5(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock5(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock5(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock5(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467block7(nn.Module):
    #block结构消融实验，对应表格第八行，即stride=2与stride=1时都使用双分支结构，同时去掉第一个1*1卷积
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DPEblock6(128, 128, dilations1, 1, 1, attention),
            DPEblock6(128, 128, dilations1, 1, 1, attention),
            DPEblock6(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DPEblock6(256, 256, dilations1, 1, 1, attention),
            DPEblock6(256, 256, dilations1, 1, 1, attention),
            DPEblock6(256, 256, dilations1, 1, 1, attention),
            DPEblock6(256, 256, dilations1, 1, 1, attention),
            DPEblock6(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock6(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock6(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock6(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock6(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock6(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock6(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d2(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=2
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d2244466(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=2
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, 2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, 2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d6(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=6
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d2224468(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=2
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, 2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, 2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num444(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )


    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)


        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_4}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num566(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_6}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat17(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_1,x32_7),dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat147(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_1,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_7),dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat1357(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_1,x32_3),dim=1)
        x32_cat = torch.cat((x32_cat,x32_5),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467_Yblock(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            YBlock(64, 128, dilations1, 1, 2),
            YBlock(128, 128, dilations1, 1, 1),
            YBlock(128, 128, dilations1, 1, 1),
            YBlock(128, 128, dilations1, 1, 1),
        )
        self.stage16=nn.Sequential(
            YBlock(128, 256, dilations1, 1, 2),
            YBlock(256, 256, dilations1, 1, 1),
            YBlock(256, 256, dilations1, 1, 1),
            YBlock(256, 256, dilations1, 1, 1),
            YBlock(256, 256, dilations1, 1, 1),
            YBlock(256, 256, dilations1, 1, 1)

        )

        self.stage32_1 = nn.Sequential(
            YBlock(256, 256, dilations2, 1, 2)
        )
        self.stage32_2 = nn.Sequential(
            YBlock(256, 256, dilations2, 1, 1)
        )
        self.stage32_3 = nn.Sequential(
            YBlock(256, 256, dilations2, 1, 1)
        )
        self.stage32_4 = nn.Sequential(
            YBlock(256, 256, dilations2, 1, 1)
        )
        self.stage32_5 = nn.Sequential(
            YBlock(256, 256, dilations2, 1, 1)
        )
        self.stage32_6 = nn.Sequential(
            YBlock(256, 256, dilations2, 1, 1)
        )
        self.stage32_7 = nn.Sequential(
            YBlock(256, 256, dilations2, 1, 1)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467_RCDblock(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DBlock(64, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], 16, 2),
            DBlock(256, 256, [1], 16, 1),
            DBlock(256, 256, [1], 16, 1),
            DBlock(256, 256, [1], 16, 1),
            DBlock(256, 256, [1], 16, 1),
            DBlock(256, 256, [1], 16, 1)

        )

        self.stage32_1 = nn.Sequential(
            DBlock(256, 256, [4], 16, 2)
        )
        self.stage32_2 = nn.Sequential(
            DBlock(256, 256, [4], 16, 1)
        )
        self.stage32_3 = nn.Sequential(
            DBlock(256, 256, [4], 16, 1)
        )
        self.stage32_4 = nn.Sequential(
            DBlock(256, 256, [4], 16, 1)
        )
        self.stage32_5 = nn.Sequential(
            DBlock(256, 256, [4], 16, 1)
        )
        self.stage32_6 = nn.Sequential(
            DBlock(256, 256, [4], 16, 1)
        )
        self.stage32_7 = nn.Sequential(
            DBlock(256, 256, [4], 16, 1)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class CDBlock_0(nn.Module): ########### ICANet block
    def __init__(self, in_channels, out_channels, dilations, stride):
        super().__init__()
        self.stride = stride
        dilation=dilations[0]

        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        #self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=1,groups=out_channels, padding=dilation,dilation=dilation,bias=False)
        #self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        x=self.conv1(x)
        #x=self.bn1(x)
        x = self.act1(x)
        if self.stride == 2:
            x=self.avg(x)
        x=self.conv2(x)
        #x=self.bn2(x)
        x=self.act2(x)
        return x

class RegSegBody_17_wh17num467d4cat2467_IDSblock(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            CDBlock_0(64, 128, [1],2),
            CDBlock_0(128, 128, [1], 1),
            CDBlock_0(128, 128, [1], 1),
            CDBlock_0(128, 128, [1], 1),
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1], 2),
            CDBlock_0(256, 256, [1], 1),
            CDBlock_0(256, 256, [1], 1),
            CDBlock_0(256, 256, [1], 1),
            CDBlock_0(256, 256, [1], 1),
            CDBlock_0(256, 256, [1], 1)

        )

        self.stage32_1 = nn.Sequential(
            CDBlock_0(256, 256, [4], 2)
        )
        self.stage32_2 = nn.Sequential(
            CDBlock_0(256, 256, [4], 1)
        )
        self.stage32_3 = nn.Sequential(
            CDBlock_0(256, 256, [4], 1)
        )
        self.stage32_4 = nn.Sequential(
            CDBlock_0(256, 256, [4], 1)
        )
        self.stage32_5 = nn.Sequential(
            CDBlock_0(256, 256, [4], 1)
        )
        self.stage32_6 = nn.Sequential(
            CDBlock_0(256, 256, [4], 1)
        )
        self.stage32_7 = nn.Sequential(
            CDBlock_0(256, 256, [4], 1)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467_channel1(nn.Module): #修改block数量 修改通道数为（32，64，64，128，256）
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 64, dilations1, 1, 2, attention),
            DDWblock(64, 64, dilations1, 1, 1, attention),
            DDWblock(64, 64, dilations1, 1, 1, attention),
            DDWblock(64, 64, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467_channel2(nn.Module): #修改block数量 修改通道数为（32，64，128，128，256）
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467_channel4(nn.Module): #修改block数量 修改通道数为（32，64，128，128，256）
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 512, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467_channel5(nn.Module): #修改block数量 修改通道数为（32，64，128，256，1024）
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 1024, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(1024, 1024, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(1024, 1024, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(1024, 1024, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(1024, 1024, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(1024, 1024, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(1024, 1024, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)



        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num345d4cat1245(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )




    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)

        x32_cat = torch.cat((x32_1,x32_2),dim=1)
        x32_cat = torch.cat((x32_cat,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_5), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num665(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )




    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)


        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_5}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num11244(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.stage2=nn.Sequential(
            #ConvBnAct(3 , 32, 3, 2, 1)
            DDWblock(3, 32, dilations1, 1, 2, attention),
        )

        self.stage4=nn.Sequential(
            #ConvBnAct(32, 64, 3, 2, 1)
            DDWblock(32, 64, dilations1, 1, 2, attention),
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )



    def forward(self,x):
        x2=self.stage2(x)
        x4=self.stage4(x2)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)


        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_4}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num11357(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.stage2 = nn.Sequential(
            # ConvBnAct(3 , 32, 3, 2, 1)
            DDWblock(3, 32, dilations1, 1, 2, attention),
        )

        self.stage4=nn.Sequential(
            DDWblock(32, 64, dilations1, 1, 2, attention),
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )




    def forward(self,x):
        x2 = self.stage2(x)
        x4=self.stage4(x2)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_5(x32_5)
        x32_7 = self.stage32_5(x32_6)


        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num01467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.stage2 = nn.Sequential(
            ConvBnAct(3 ,32, 3, 2, 1)
            #DDWblock(3, 32, dilations1, 1, 2, attention),
        )

        self.stage4=nn.Sequential(
            DDWblock(32, 64, dilations1, 1, 2, attention),
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )





    def forward(self,x):
        x2 = self.stage2(x)
        x4=self.stage4(x2)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_5(x32_5)
        x32_7 = self.stage32_5(x32_6)

        x32_cat = torch.cat((x32_2, x32_4), dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)


        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num11467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.stage2 = nn.Sequential(
            #ConvBnAct(3 ,32, 3, 2, 1)
            DDWblock(3, 32, dilations1, 1, 2, attention),
        )

        self.stage4=nn.Sequential(
            DDWblock(32, 64, dilations1, 1, 2, attention),
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )




    def forward(self,x):
        x2 = self.stage2(x)
        x4=self.stage4(x2)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_5(x32_5)
        x32_7 = self.stage32_5(x32_6)

        x32_cat = torch.cat((x32_2, x32_4), dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)


        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num11345(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.stage2 = nn.Sequential(
            # ConvBnAct(3 , 32, 3, 2, 1)
            DDWblock(3, 32, dilations1, 1, 2, attention),
        )

        self.stage4=nn.Sequential(
            DDWblock(32, 64, dilations1, 1, 2, attention),
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )

    def forward(self,x):
        x2 = self.stage2(x)
        x4=self.stage4(x2)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)


        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_5}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num01345(nn.Module):  # 修改block数量
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'
        self.stage2 = nn.Sequential(
            ConvBnAct(3 , 32, 3, 2, 1)
            #DDWblock(3, 32, dilations1, 1, 2, attention),
        )

        self.stage4 = nn.Sequential(
            DDWblock(32, 64, dilations1, 1, 2, attention),
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 1, attention)
        )

    def forward(self, x):
        x2 = self.stage2(x)
        x4 = self.stage4(x2)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)
        x32_5 = self.stage32_5(x32_4)

        # x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        # x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32_5}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}
        # return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d28cat(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.stage2 = nn.Sequential(
            ConvBnAct(3 , 32, 3, 2, 1)
        )

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, 8, 1, 2, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, 12, 1, 2, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, 16, 1, 2, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, 20, 1, 2, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, 24, 1, 2, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, 28, 1, 2, attention)
        )



    def forward(self,x):
        x2=self.stage2(x)
        x4=self.stage4(x2)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x16)
        x32_3=self.stage32_3(x16)
        x32_4=self.stage32_4(x16)
        x32_5=self.stage32_5(x16)
        x32_6 = self.stage32_6(x16)
        x32_7 = self.stage32_7(x16)

        x32_cat = torch.cat((x32_1,x32_2),dim=1)
        x32_cat = torch.cat((x32_cat,x32_3),dim=1)
        x32_cat = torch.cat((x32_cat, x32_4), dim=1)
        x32_cat = torch.cat((x32_cat, x32_5), dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1792}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4add2467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.add(x32_2,x32_4)
        x32_cat = torch.add(x32_cat,x32_6)
        x32_cat = torch.add(x32_cat, x32_7)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num11467d4cat2467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            DDWblock(32, 64, dilations1, 1, 2, attention),
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )




    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_5(x32_5)
        x32_7 = self.stage32_5(x32_6)

        x32_cat = torch.cat((x32_2, x32_4), dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num689d4cat24679(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_9 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_7(x32_7)
        x32_9 = self.stage32_7(x32_8)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        x32_cat = torch.cat((x32_cat, x32_9), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num7910d4cat2467910(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_9 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_10 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_7(x32_7)
        x32_9 = self.stage32_7(x32_8)
        x32_10 = self.stage32_7(x32_9)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        x32_cat = torch.cat((x32_cat, x32_9), dim=1)
        x32_cat = torch.cat((x32_cat, x32_10), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num689d4cat24679512(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 512, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )
        self.stage32_9 = nn.Sequential(
            DDWblock(512, 512, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_7(x32_7)
        x32_9 = self.stage32_7(x32_8)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        x32_cat = torch.cat((x32_cat, x32_9), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num3689d4cat24679(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1),
            ConvBnAct(64, 64, 3, 2, 1),
            ConvBnAct(64, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_9 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_7(x32_7)
        x32_9 = self.stage32_7(x32_8)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        x32_cat = torch.cat((x32_cat, x32_9), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num7912d4cat2467912(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_9 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_10 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_11 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_12 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_7(x32_7)
        x32_9 = self.stage32_7(x32_8)
        x32_10 = self.stage32_7(x32_9)
        x32_11 = self.stage32_7(x32_10)
        x32_12 = self.stage32_7(x32_11)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        x32_cat = torch.cat((x32_cat, x32_9), dim=1)
        x32_cat = torch.cat((x32_cat, x32_12), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d8cat2467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=8
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d10cat2467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=10
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num7912d8cat2467912(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=8
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_9 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_10 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_11 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_12 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_7(x32_7)
        x32_9 = self.stage32_7(x32_8)
        x32_10 = self.stage32_7(x32_9)
        x32_11 = self.stage32_7(x32_10)
        x32_12 = self.stage32_7(x32_11)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        x32_cat = torch.cat((x32_cat, x32_9), dim=1)
        x32_cat = torch.cat((x32_cat, x32_12), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num7912d4cat2467913(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_9 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_10 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_11 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_12 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_13 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_7(x32_7)
        x32_9 = self.stage32_7(x32_8)
        x32_10 = self.stage32_7(x32_9)
        x32_11 = self.stage32_7(x32_10)
        x32_12 = self.stage32_7(x32_11)
        x32_13 = self.stage32_7(x32_12)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        x32_cat = torch.cat((x32_cat, x32_9), dim=1)
        x32_cat = torch.cat((x32_cat, x32_13), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num79121d4cat2467913(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_9 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_10 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_11 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_12 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_13 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_7(x32_7)
        x32_9 = self.stage32_7(x32_8)
        x32_10 = self.stage32_7(x32_9)
        x32_11 = self.stage32_7(x32_10)
        x32_12 = self.stage32_7(x32_11)
        x32_13 = self.stage32_7(x32_12)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        x32_cat = torch.cat((x32_cat, x32_9), dim=1)
        x32_cat = torch.cat((x32_cat, x32_13), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_whregsegnum467d4cat2467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1= [1]
        dilations2 =[1,2]
        dilations3 = [1,4]
        dilations4 = [1,14]
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DBlock_new(64, 128, dilations2, gw, 2, attention),
            DBlock_new(128, 128, dilations2, gw, 1, attention),
            DBlock_new(128, 128, dilations2, gw, 1, attention),
            DBlock_new(128, 128, dilations2, gw, 1, attention),
        )
        self.stage16=nn.Sequential(
            DBlock_new(128, 256, dilations3, gw, 2, attention),
            DBlock_new(256, 256, dilations3, gw, 1, attention),
            DBlock_new(256, 256, dilations3, gw, 1, attention),
            DBlock_new(256, 256, dilations3, gw, 1, attention),
            DBlock_new(256, 256, dilations3, gw, 1, attention),
            DBlock_new(256, 256, dilations3, gw, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DBlock_new(256, 256, dilations4, gw, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DBlock_new(256, 256, dilations4, gw, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DBlock_new(256, 256, dilations4, gw, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DBlock_new(256, 256, dilations4, gw, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DBlock_new(256, 256, dilations4, gw, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DBlock_new(256, 256, dilations4, gw, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DBlock_new(256, 256, dilations4, gw, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467sb(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock1(64, 128, dilations1, 1, 2, attention),
            DPEblock1(128, 128, dilations1, 1, 1, attention),
            DPEblock1(128, 128, dilations1, 1, 1, attention),
            DPEblock1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock1(128, 256, dilations1, 1, 2, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention),
            DPEblock1(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock1(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467sbdw1(nn.Module): #block为单分支，同时去掉dS后的1x1
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock5(64, 128, dilations1, 1, 2, attention),
            DPEblock5(128, 128, dilations1, 1, 1, attention),
            DPEblock5(128, 128, dilations1, 1, 1, attention),
            DPEblock5(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock5(128, 256, dilations1, 1, 2, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention),
            DPEblock5(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock5(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock5(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467dbdw1(nn.Module): #block为单分支，同时去掉dS后的1x1
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock4(64, 128, dilations1, 1, 2, attention),
            DPEblock4(128, 128, dilations1, 1, 1, attention),
            DPEblock4(128, 128, dilations1, 1, 1, attention),
            DPEblock4(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock4(128, 256, dilations1, 1, 2, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention),
            DPEblock4(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock4(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock4(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock4(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock4(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock4(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock4(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock4(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467sbdwhou1(nn.Module): #block为单分支，同时去掉dS后的1x1
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock7(64, 128, dilations1, 1, 2, attention),
            DPEblock7(128, 128, dilations1, 1, 1, attention),
            DPEblock7(128, 128, dilations1, 1, 1, attention),
            DPEblock7(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock7(128, 256, dilations1, 1, 2, attention),
            DPEblock7(256, 256, dilations1, 1, 1, attention),
            DPEblock7(256, 256, dilations1, 1, 1, attention),
            DPEblock7(256, 256, dilations1, 1, 1, attention),
            DPEblock7(256, 256, dilations1, 1, 1, attention),
            DPEblock7(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock7(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock7(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock7(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock7(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock7(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock7(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock7(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467db1dw(nn.Module): #block为单分支，同时去掉dS后的1x1
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DPEblock8(64, 128, dilations1, 1, 2, attention),
            DPEblock8(128, 128, dilations1, 1, 1, attention),
            DPEblock8(128, 128, dilations1, 1, 1, attention),
            DPEblock8(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DPEblock8(128, 256, dilations1, 1, 2, attention),
            DPEblock8(256, 256, dilations1, 1, 1, attention),
            DPEblock8(256, 256, dilations1, 1, 1, attention),
            DPEblock8(256, 256, dilations1, 1, 1, attention),
            DPEblock8(256, 256, dilations1, 1, 1, attention),
            DPEblock8(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DPEblock8(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DPEblock8(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DPEblock8(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DPEblock8(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DPEblock8(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DPEblock8(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DPEblock8(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d4cat2467posi(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblockp(64, 128, dilations1, 1, 2, attention),
            DDWblockp(128, 128, dilations1, 1, 1, attention),
            DDWblockp(128, 128, dilations1, 1, 1, attention),
            DDWblockp(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblockp(128, 256, dilations1, 1, 2, attention),
            DDWblockp(256, 256, dilations1, 1, 1, attention),
            DDWblockp(256, 256, dilations1, 1, 1, attention),
            DDWblockp(256, 256, dilations1, 1, 1, attention),
            DDWblockp(256, 256, dilations1, 1, 1, attention),
            DDWblockp(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d1cat2467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=1
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num467d2244466cat2467(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=1
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, 2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(256, 256, 2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, 4, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(256, 256, 6, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17num7912d4cat2467912posi(nn.Module): #修改block数量
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

        )

        self.stage32_1 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_8 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_9 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_10 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_11 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_12 = nn.Sequential(
            DDWblockp(256, 256, dilations2, 1, 1, attention)
        )



    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_6 = self.stage32_6(x32_5)
        x32_7 = self.stage32_7(x32_6)
        x32_8 = self.stage32_7(x32_7)
        x32_9 = self.stage32_7(x32_8)
        x32_10 = self.stage32_7(x32_9)
        x32_11 = self.stage32_7(x32_10)
        x32_12 = self.stage32_7(x32_11)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat,x32_6),dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        x32_cat = torch.cat((x32_cat, x32_9), dim=1)
        x32_cat = torch.cat((x32_cat, x32_12), dim=1)

        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        #return {"4":48,"8":128,"16":256,"32":256}