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

class DDWblocklight1(nn.Module):  # wh
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
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=in_channels,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.bn2 = norm2d(out_channels)  # BN
        self.act2 = activation()  # RELU
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
        x = self.act2(x + shortcut)
        return x

class DDWblocklight2(nn.Module):  # wh
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
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        # if self.se is not None:
        #     x=self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DDWblocklight3(nn.Module):  # wh
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
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
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
        # if self.se is not None:
        #     x=self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x


class DDWblock2(nn.Module):  # wh
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


class PMSA(nn.Module):
    def __init__(self, in_channels, channels):
        super(PMSA, self).__init__(
            )
        self.query_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.key_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.value_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        #print("*********query's shape***********",query.shape)#[1,2048,128]

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)

        key = key.reshape(*key.shape[:2], -1)
        #print("***********key's shape***********",key.shape)
        value = value.reshape(*value.shape[:2], -1)
        #print("***********value's reshape shape***********", value.shape)
        value = value.permute(0, 2, 1).contiguous()
        #print("***********value's permute shape***********", value.shape)

        sim_map = torch.matmul(query, key)
        #print("**********sim_map's shape************", sim_map.shape)#[1,2048,2048]

        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        #print("***********context's shape***********", context.shape)
        context = context.permute(0, 2, 1).contiguous()
        #print("***********context's permute shape***********", context.shape)
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        #print("***********context's reshape shape***********", context.shape)

        return context

class PMSA_DNL(nn.Module):
    def __init__(self, in_channels, channels):
        super(PMSA_DNL, self).__init__(
            )
        self.query_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.key_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.value_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.theta = nn.Conv2d(256, 32, kernel_size=1)
        self.phi = nn.Conv2d(256, 32, kernel_size=1)
        self.g = nn.Conv2d(256, 32, kernel_size=1)
        self.conv_out = nn.Conv2d(32, 128, kernel_size=1)

    def forward(self, x):
        """Forward function."""
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, 32, -1)
        theta_x = self.theta(x).view(batch_size, 32, -1)
        phi_x = self.phi(x).view(batch_size, 32, -1).permute(0, 2, 1)

        theta_x = F.softmax(theta_x, dim=-1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous().view(batch_size, 32, *x.size()[2:])
        output = self.conv_out(y)

        return output

class PMSA_regseg(nn.Module):
    def __init__(self, in_channels, channels):
        super(PMSA_regseg, self).__init__(
            # key_in_channels=in_channels + channels,  # =2048+512=2560
            # query_in_channels=in_channels,  # 2048
            # channels=channels,  # 512
            # out_channels=in_channels,  # 2048
            # share_key_query=False,
            # query_downsample=None,
            # key_downsample=None,
            # key_query_num_convs=1,
            # key_query_norm=False,
            # value_out_num_convs=1,
            # value_out_norm=False,
            # matmul_norm=False,
            # with_out=False,
            # conv_cfg=conv_cfg,
            # norm_cfg=norm_cfg,
            # act_cfg=act_cfg
            )
        self.query_project = ConvBnAct(
            in_channels=320,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.key_project = ConvBnAct(
            in_channels=320,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.value_project = ConvBnAct(
            in_channels=320,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        #print("*********query's shape***********",query.shape)#[1,2048,128]

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)

        key = key.reshape(*key.shape[:2], -1)
        #print("***********key's shape***********",key.shape)
        value = value.reshape(*value.shape[:2], -1)
        #print("***********value's reshape shape***********", value.shape)
        value = value.permute(0, 2, 1).contiguous()
        #print("***********value's permute shape***********", value.shape)

        sim_map = torch.matmul(query, key)
        #print("**********sim_map's shape************", sim_map.shape)#[1,2048,2048]

        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        #print("***********context's shape***********", context.shape)
        context = context.permute(0, 2, 1).contiguous()
        #print("***********context's permute shape***********", context.shape)
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        #print("***********context's reshape shape***********", context.shape)

        return context

class RegSegBody_17_wh19_self_attention(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage4内部进行两次下采样操作
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,

            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.conv1 = ConvBnAct(1536, 256, 1)

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 2, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
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

        x32 = x32_7

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x16
        x_key = x32
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class PMSA_cat(nn.Module):
    def __init__(self, in_channels, channels):
        super(PMSA_cat, self).__init__(
            # key_in_channels=in_channels + channels,  # =2048+512=2560
            # query_in_channels=in_channels,  # 2048
            # channels=channels,  # 512
            # out_channels=in_channels,  # 2048
            # share_key_query=False,
            # query_downsample=None,
            # key_downsample=None,
            # key_query_num_convs=1,
            # key_query_norm=False,
            # value_out_num_convs=1,
            # value_out_norm=False,
            # matmul_norm=False,
            # with_out=False,
            # conv_cfg=conv_cfg,
            # norm_cfg=norm_cfg,
            # act_cfg=act_cfg
            )
        self.query_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.key_project = ConvBnAct(
            in_channels=1024,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.value_project = ConvBnAct(
            in_channels=1024,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        #print("*********query's shape***********",query.shape)#[1,2048,128]

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)

        key = key.reshape(*key.shape[:2], -1)
        #print("***********key's shape***********",key.shape)
        value = value.reshape(*value.shape[:2], -1)
        #print("***********value's reshape shape***********", value.shape)
        value = value.permute(0, 2, 1).contiguous()
        #print("***********value's permute shape***********", value.shape)

        sim_map = torch.matmul(query, key)
        #print("**********sim_map's shape************", sim_map.shape)#[1,2048,2048]

        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        #print("***********context's shape***********", context.shape)
        context = context.permute(0, 2, 1).contiguous()
        #print("***********context's permute shape***********", context.shape)
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        #print("***********context's reshape shape***********", context.shape)

        return context

def activation1():
    return nn.ReLU(inplace=False)

class PMSA_vit1(nn.Module):
    def __init__(self, in_channels, channels):
        super(PMSA_vit1, self).__init__(

            )
        self.query_project = ConvBnAct(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.key_project = ConvBnAct(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.value_project = ConvBnAct(
            in_channels=256,
            out_channels=256,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.dim=8
        self.act = activation()
        self.act1 = activation1()
        self.act2 = nn.GELU()
        self.eps =1.0e-15


    def forward(self, query_feats, key_feats, value_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        key = self.key_project(key_feats)
        value = self.value_project(value_feats)
        H = value.shape[2]
        W = value.shape[3]

        q = torch.reshape(
            query,
            (
                batch_size,
                -1,
                self.dim,
                query.shape[2]*query.shape[3],
            ),
        )
        #print("***********context's q shape***********", q.shape)

        k = torch.reshape(
            key,
            (
                batch_size,
                -1,
                self.dim,
                key.shape[2]*key.shape[3],
            ),
        )
        #print("***********context's k shape***********", k.shape)

        v = torch.reshape(
            value,
            (
                batch_size,
                -1,
                self.dim,
                value.shape[2]*value.shape[3],
            ),
        )
        #print("***********context's v shape***********", v.shape)

        q = self.act2(q)
        k = self.act2(k)
        #v = F.pad(v, (0, 1), mode="constant", value=1)

        trans_v = v.transpose(-1,-2)
        trans_q = q.transpose(-1,-2)
        #print("***********context's trans_v shape***********", trans_v.shape)
        kv = torch.matmul(k,trans_v)
        #print("***********context's kv shape***********", kv.shape)

        #kv_map = F.softmax(kv, dim=-1)

        out = torch.matmul(trans_q,kv)
        #out = out[..., :-1] / (out[..., -1:] + self.eps)
        #print("***********context's out1 shape***********", out.shape)
        out = torch.transpose(out,-1,-2)
        #print("***********context's out2 shape***********", out.shape)
        out = torch.reshape(out,(batch_size,-1,H,W))
        #print("***********context's out3 shape***********", out.shape)

        #out = torch.reshape(out(batch_size,-1,H))


        #query = query.reshape(*query.shape[:2], -1)
        #query = query.permute(0, 2, 1).contiguous()
        #print("*********query's shape***********",query.shape)#[1,2048,128]

        #key = self.key_project(key_feats)
        #value = self.value_project(key_feats)

        #key = key.reshape(*key.shape[:2], -1)
        #print("***********key's shape***********",key.shape)
        #value = value.reshape(*value.shape[:2], -1)
        #print("***********value's reshape shape***********", value.shape)
        #value = value.permute(0, 2, 1).contiguous()
        #print("***********value's permute shape***********", value.shape)

        #sim_map = torch.matmul(query, key)
        #print("**********sim_map's shape************", sim_map.shape)#[1,2048,2048]

        #sim_map = F.softmax(sim_map, dim=-1)

        #context = torch.matmul(sim_map, value)
        #print("***********context's shape***********", context.shape)
        #context = context.permute(0, 2, 1).contiguous()
        #print("***********context's permute shape***********", context.shape)
        #context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        #print("***********context's reshape shape***********", context.shape)

        return out

class PMSA_vit2(nn.Module):
    def __init__(self, in_channels, channels):
        super(PMSA_vit2, self).__init__(
            # key_in_channels=in_channels + channels,  # =2048+512=2560
            # query_in_channels=in_channels,  # 2048
            # channels=channels,  # 512
            # out_channels=in_channels,  # 2048
            # share_key_query=False,
            # query_downsample=None,
            # key_downsample=None,
            # key_query_num_convs=1,
            # key_query_norm=False,
            # value_out_num_convs=1,
            # value_out_norm=False,
            # matmul_norm=False,
            # with_out=False,
            # conv_cfg=conv_cfg,
            # norm_cfg=norm_cfg,
            # act_cfg=act_cfg

            )
        self.query_project = ConvBnAct(
            in_channels=256,
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.key_project = ConvBnAct(
            in_channels=256,
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.value_project = ConvBnAct(
            in_channels=512,
            out_channels=512,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.dim=8
        self.act = activation()
        self.act1 = activation1()
        self.eps =1.0e-15


    def forward(self, query_feats, key_feats, value_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        key = self.key_project(key_feats)
        value = self.value_project(value_feats)
        H = value.shape[2]
        W = value.shape[3]

        q = torch.reshape(
            query,
            (
                batch_size,
                -1,
                2*self.dim,
                query.shape[2]*query.shape[3],
            ),
        )
        #print("***********context's q shape***********", q.shape)

        k = torch.reshape(
            key,
            (
                batch_size,
                -1,
                2 * self.dim,
                key.shape[2]*key.shape[3],
            ),
        )
        #print("***********context's k shape***********", k.shape)

        v = torch.reshape(
            value,
            (
                batch_size,
                -1,
                512,
                #2 * self.dim,
                value.shape[2]*value.shape[3],
            ),
        )
        #print("***********context's v shape***********", v.shape)

        q = self.act(q)
        k = self.act(k)
        v = F.pad(v, (0, 1), mode="constant", value=1)

        trans_v = v.transpose(-1,-2)
        trans_q = q.transpose(-1,-2)
        #print("***********context's trans_v shape***********", trans_v.shape)
        kv = torch.matmul(k,trans_v)
        #print("***********context's kv shape***********", kv.shape)

        #kv_map = F.softmax(kv, dim=-1)

        out = torch.matmul(trans_q,kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        #print("***********context's out1 shape***********", out.shape)
        out = torch.transpose(out,-1,-2)
        #print("***********context's out2 shape***********", out.shape)
        out = torch.reshape(out,(batch_size,-1,H,W))
        #print("***********context's out3 shape***********", out.shape)

        #out = torch.reshape(out(batch_size,-1,H))


        #query = query.reshape(*query.shape[:2], -1)
        #query = query.permute(0, 2, 1).contiguous()
        #print("*********query's shape***********",query.shape)#[1,2048,128]

        #key = self.key_project(key_feats)
        #value = self.value_project(key_feats)

        #key = key.reshape(*key.shape[:2], -1)
        #print("***********key's shape***********",key.shape)
        #value = value.reshape(*value.shape[:2], -1)
        #print("***********value's reshape shape***********", value.shape)
        #value = value.permute(0, 2, 1).contiguous()
        #print("***********value's permute shape***********", value.shape)

        #sim_map = torch.matmul(query, key)
        #print("**********sim_map's shape************", sim_map.shape)#[1,2048,2048]

        #sim_map = F.softmax(sim_map, dim=-1)

        #context = torch.matmul(sim_map, value)
        #print("***********context's shape***********", context.shape)
        #context = context.permute(0, 2, 1).contiguous()
        #print("***********context's permute shape***********", context.shape)
        #context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        #print("***********context's reshape shape***********", context.shape)

        return out

class RegSegBody_17_wh19_self_attention_vit(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage4内部进行两次下采样操作  stage5内部将1，3，5，7block的特征进行cat操作
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA_vit1(
            1024,
            128,
            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        x32 = torch.cat((x32_2,x32_4),dim=1)
        x32 = torch.cat((x32,x32_6),dim=1)
        x32 = torch.cat((x32,x32_7),dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x32_1
        x_key = x32_4
        x_value = x32_7
        x_self_attention = self.attention(x_query, x_key, x_value)
        x_out = x32_7 + x_self_attention
        #x_self_attention_vit = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_out}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 1152}

class RegSegBody_17_wh19_self_attention_vit234(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage4内部进行两次下采样操作  stage5内部将1，3，5，7block的特征进行cat操作
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA_vit1(
            1024,
            128,
            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
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

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage64_2 = nn.Sequential(
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

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_1(x64_1)

        x32 = torch.cat((x32_2,x32_4),dim=1)
        x32 = torch.cat((x32,x32_6),dim=1)
        x32 = torch.cat((x32,x32_7),dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_value = x64_2
        x_self_attention = self.attention(x_query, x_key, x_value)
        x_out = x64_2 + x_self_attention
        #x_self_attention_vit = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_out}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 1152}

class PMSA_selfvit(nn.Module):
    def __init__(self, in_channels, channels):
        super(PMSA, self).__init__(
            # key_in_channels=in_channels + channels,  # =2048+512=2560
            # query_in_channels=in_channels,  # 2048
            # channels=channels,  # 512
            # out_channels=in_channels,  # 2048
            # share_key_query=False,
            # query_downsample=None,
            # key_downsample=None,
            # key_query_num_convs=1,
            # key_query_norm=False,
            # value_out_num_convs=1,
            # value_out_norm=False,
            # matmul_norm=False,
            # with_out=False,
            # conv_cfg=conv_cfg,
            # norm_cfg=norm_cfg,
            # act_cfg=act_cfg
            )
        self.query_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.key_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.value_project = ConvBnAct(
            in_channels=256,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.dim = 8

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()
        print("*********query's shape***********",query.shape)#[1,2048,128]

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)

        key = key.reshape(*key.shape[:2], -1)
        print("***********key's shape***********",key.shape)
        value = value.reshape(*value.shape[:2], -1)
        print("***********value's reshape shape***********", value.shape)
        value = value.permute(0, 2, 1).contiguous()
        print("***********value's permute shape***********", value.shape)

        sim_map = torch.matmul(query, key)
        print("**********sim_map's shape************", sim_map.shape)#[1,2048,2048]

        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        print("***********context's shape***********", context.shape)
        context = context.permute(0, 2, 1).contiguous()
        print("***********context's permute shape***********", context.shape)
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        print("***********context's reshape shape***********", context.shape)

        return context


class RegSegBody_17_wh19_self_attention_cat(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage4内部进行两次下采样操作  stage5内部将1，3，5，7block的特征进行cat操作
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA_vit(
            1024,
            128,

            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        #x32 = torch.cat((x32_2,x32_4),dim=1)
        #x32 = torch.cat((x32,x32_6),dim=1)
        #x32 = torch.cat((x32,x32_7),dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x32_1
        x_key = x32_4
        x_value = x32_7
        x_self_attention = self.attention(x_query, x_key, x_value)
        x_self_attention_vit = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 1152}

class RegSegBody_17_wh19_self_attention_2(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,

            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        x32 = x32_7

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x32_1
        x_key = x32
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_3(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    #相比之前增加block数量
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,

            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1),
            ConvBnAct(64, 64, 3, 1, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
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
        x32_8 = self.stage32_8(x32_7)
        x32_9 = self.stage32_9(x32_8)

        x32 = x32_9

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x32_1
        x_key = x32
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 64, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_4(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,

            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        x32 = x32_7

        # 添加交叉分辨率注意力，在x16和x32_1之间，
        x_query = x32_1
        x_key = x32
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_cat_stage3(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage4内部进行1次下采样操作  stage5内部将2，4，6，7block的特征进行cat操作
    # 将stage3的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.stage8_conv1=ConvBnAct(128,256,3,2,1)
        self.stage8_conv2=ConvBnAct(256,256,3,2,1)

        self.attention = PMSA_cat(
            1024,
            128,

            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        x32 = torch.cat((x32_2,x32_4),dim=1)
        x32 = torch.cat((x32,x32_6),dim=1)
        x32 = torch.cat((x32,x32_7),dim=1)

        x8_final = self.stage8_conv1(x8)
        x8_final = self.stage8_conv2(x8_final)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x8_final
        x_key = x32
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 1152}

class RegSegBody_17_wh19_self_attention_selfvit(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,

            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        x32 = x32_7

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x32_1
        x_key = x32
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s6(nn.Module): #部署stage6的block数量为3
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage64_3 = nn.Sequential(
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

        x64_1 = self.stage64_1(x32_6)
        x64_2 = self.stage64_2(x64_1)
        x64_3 = self.stage64_3(x64_2)

        x64 = torch.cat((x64_1, x64_2), dim=1)
        x64 = torch.cat((x64, x64_3), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x64],dim=1)
        return {"4": x8, "8": x16, "16": x32_6, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s613qk(nn.Module): #部署stage6的block数量为3
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage64_3 = nn.Sequential(
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

        x64_1 = self.stage64_1(x32_6)
        x64_2 = self.stage64_2(x64_1)
        x64_3 = self.stage64_3(x64_2)

        x64 = torch.cat((x64_1, x64_3), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_3
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x64],dim=1)
        return {"4": x8, "8": x16, "16": x32_6, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s6_2(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
        )

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)

        x64_1 = self.stage64_1(x32)
        x64_2 = self.stage64_2(x64_1)

        x64 = torch.cat((x64_1, x64_2), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x64],dim=1)
        return {"4": x8, "8": x16, "16": x32, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5b7s6b2_cat12(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)

        x64_1 = self.stage64_1(x32)
        x64_2 = self.stage64_2(x64_1)

        x64 = torch.cat((x64_1, x64_2), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x64],dim=1)
        return {"4": x8, "8": x16, "16": x32, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s6_2right_1q3kv(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
        )

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)

        x64_1 = self.stage64_1(x32)
        x64_2 = self.stage64_2(x64_1)
        x64_3 = self.stage64_2(x64_2)

        #x64 = torch.cat((x64_1, x64_2), dim=1)

        x_query = x64_1
        x_key = x64_3
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x64_3],dim=1)
        return {"4": x8, "8": x16, "16": x32, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
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

        x64_1 = self.stage64_1(x32_7)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_7), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_7.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        x32 = torch.cat((x32_1, x32_7), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x32_1
        x_key = x32_7
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s6_3(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,

            # in_channels=self.in_channels,
            # channels=self.channels,
            #conv_cfg=self.conv_cfg,
            #norm_cfg=self.norm_cfg,
            #act_cfg=self.act_cfg
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
        )

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)

        x64_1 = self.stage64_1(x32)
        x64_2 = self.stage64_2(x64_1)
        x64_3 = self.stage64_3(x64_2)

        x64 = torch.cat((x64_1, x64_2), dim=1)
        x64 = torch.cat((x64, x64_3), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_3
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x64],dim=1)
        return {"4": x8, "8": x16, "16": x32, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s6_2right(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
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

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
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

        x64_1 = self.stage64_1(x32_7)
        x64_2 = self.stage64_2(x64_1)

        x64 = torch.cat((x64_1, x64_2), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x64],dim=1)
        return {"4": x8, "8": x16, "16": x32_7, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b3342(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
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

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
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

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b3342d6(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 6
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
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

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
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

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations1, 1, 2, attention),
            DDWblock(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
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

        self.stage64_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
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

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342dd4(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock4(64, 128, dilations1, 1, 2, attention),
            DDWblock4(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock4(128, 256, dilations1, 1, 2, attention),
            DDWblock4(256, 256, dilations1, 1, 1, attention),
            DDWblock4(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock4(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock4(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock4(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock4(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblock4(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblock4(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342dd3(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblock3(64, 128, dilations1, 1, 2, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblock3(128, 256, dilations1, 1, 2, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight2(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight2(64, 128, dilations1, 1, 2, attention),
            DDWblocklight2(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight2(128, 256, dilations1, 1, 2, attention),
            DDWblocklight2(256, 256, dilations1, 1, 1, attention),
            DDWblocklight2(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight2(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight2(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight2(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight2(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight2(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight2(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight3(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight3(64, 128, dilations1, 1, 2, attention),
            DDWblocklight3(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight3(128, 256, dilations1, 1, 2, attention),
            DDWblocklight3(256, 256, dilations1, 1, 1, attention),
            DDWblocklight3(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight3(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight3(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight3(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight3(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight3(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight3(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b3332ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)

        x64_1 = self.stage64_1(x32_3)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_3), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_3.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_3, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b1352ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
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

        x64_1 = self.stage64_1(x32_5)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_5), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_5.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_5, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit2_s5_2right_b2342ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)
        x64_3 = self.stage64_2(x64_2)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_3
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2352ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)
        x32_5 = self.stage32_4(x32_4)

        x64_1 = self.stage64_1(x32_5)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_5), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_5.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_5, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b3452ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)
        x32_5 = self.stage32_4(x32_4)

        x64_1 = self.stage64_1(x32_5)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_5), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_5.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_5, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_ASPP(nn.Module): #部署stage6的block数量为2
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'



        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.aspp1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU()
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.ReLU()
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=8, dilation=8),
            nn.ReLU()
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=12, dilation=12),
            nn.ReLU()
        )
        self.aspp5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=16, dilation=16),
            nn.ReLU()
        )
        self.output = nn.Conv2d(256 * 5, 256, kernel_size=1)

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        aspp1 = self.aspp1(x32_4)
        aspp2 = self.aspp2(x32_4)
        aspp3 = self.aspp3(x32_4)
        aspp4 = self.aspp4(x32_4)
        aspp5 = self.aspp5(x32_4)
        out = torch.cat([aspp1, aspp2, aspp3, aspp4, aspp5], dim=1)
        out = self.output(out)

        # x64_1 = self.stage64_1(x32_4)
        # x64_2 = self.stage64_2(x64_1)
        #
        # x32 = torch.cat((x32_1, x32_4), dim=1)
        #
        # x_query = x64_1
        # x_key = x64_2
        # x_self_attention = self.attention(x_query, x_key)
        # x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        # x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": out}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 256}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_ASPP_psab(nn.Module): #部署stage6的block数量为2
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'
        self.attention = PMSA(
            1280,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.aspp1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU()
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.ReLU()
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=8, dilation=8),
            nn.ReLU()
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=12, dilation=12),
            nn.ReLU()
        )
        self.aspp5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=16, dilation=16),
            nn.ReLU()
        )
        self.output = nn.Conv2d(256 * 5, 256, kernel_size=1)

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        aspp1 = self.aspp1(x32_4)
        aspp2 = self.aspp2(x32_4)
        aspp3 = self.aspp3(x32_4)
        aspp4 = self.aspp4(x32_4)
        aspp5 = self.aspp5(x32_4)
        out = torch.cat([aspp1, aspp2, aspp3, aspp4, aspp5], dim=1)
        out = self.output(out)
        x_query = out
        x_key = out
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention, out], dim=1)

        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 256}


class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1NL(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        #x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_2
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        #x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention,x64_2],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_last4stage(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)
        x64 = torch.cat((x64_1, x64_2), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x_self_attention_cat, "32": x64}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_DNL(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA_DNL(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_feat = x64_2
        x_self_attention = self.attention(x_feat)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b012342ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            DDWblocklight1(32, 64, dilations1, 1, 2, attention),
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b112342ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            DDWblocklight1(32, 64, dilations1, 1, 2, attention),
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_4.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_4, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b4322ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)

        x64_1 = self.stage64_1(x32_2)
        x64_2 = self.stage64_2(x64_1)

        x32 = torch.cat((x32_1, x32_2), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x64_1
        x_key = x64_2
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_up = F.interpolate(x_self_attention, size=x32_2.shape[-2:],mode='bilinear',align_corners=False)
        x_self_attention_cat = torch.cat([x_self_attention_up,x32],dim=1)
        return {"4": x8, "8": x16, "16": x32_2, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_s5_2right_b2342ddlight1(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x64 = torch.cat((x64_1, x64_2), dim=1)

        return {"4": x8, "8": x16, "16": x32_4, "32": x64}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_s5_2right_b2342ddlight1DA(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'
        self.reduction_ratio = 8

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        x64 = torch.cat((x64_1, x64_2), dim=1)

        x_c = self.channel_att(x64)
        x_s =self.spatial_att(x64)

        x64_ca = x64 * x_c
        x64_sa = x64 * x_s

        x64_out = x64_ca + x64_sa

        return {"4": x8, "8": x16, "16": x32_4, "32": x64_out}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_psaa(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )


    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x32 = torch.cat((x32_1, x32_4), dim=1)

        # stage4中进行两次下采样，stage5不进行下采样
        x_query = x32_4
        x_key = x32_4
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x32],dim=1)
        return {"4": x4, "8": x8, "16": x16, "32": x_self_attention_cat}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}





class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()

        self.pooling_layers = nn.ModuleList()
        for pool_size in pool_sizes:
            self.pooling_layers.append(nn.AdaptiveAvgPool2d(pool_size))

        out_channels = in_channels // len(pool_sizes)
        self.conv = nn.Conv2d(1280, in_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]  # Get input feature map size

        ppm_outputs = [x]
        for layer in self.pooling_layers:
            ppm_outputs.append(F.interpolate(layer(x), size=input_size, mode='bilinear', align_corners=True))

        ppm_output = torch.cat(ppm_outputs, dim=1)
        ppm_output = self.conv(ppm_output)

        return ppm_output

class RegSegBody_17_wh19_self_attention_selfvit_s5_2right_b2342ddlight1_PPM(nn.Module): #部署stage6的block数量为2
    # 在stage4内部进行1次下采样操作,stage5进行一次下采样，stage5的第一个block的输出作为query
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 1
        dilations2 = 4
        attention = 'se'

        self.attention = PMSA(
            256,
            128,
        )

        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
        )
        self.stage8 = nn.Sequential(
            DDWblocklight1(64, 128, dilations1, 1, 2, attention),
            DDWblocklight1(128, 128, dilations1, 1, 1, attention),
        )
        self.stage16 = nn.Sequential(
            DDWblocklight1(128, 256, dilations1, 1, 2, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
            DDWblocklight1(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.stage64_1 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 2, attention)
        )

        self.stage64_2 = nn.Sequential(
            DDWblocklight1(256, 256, dilations2, 1, 1, attention)
        )

        self.ppm = PyramidPoolingModule(in_channels=256,pool_sizes=(1,2,3,6))

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32_1 = self.stage32_1(x16)
        x32_2 = self.stage32_2(x32_1)
        x32_3 = self.stage32_3(x32_2)
        x32_4 = self.stage32_4(x32_3)

        x64_1 = self.stage64_1(x32_4)
        x64_2 = self.stage64_2(x64_1)

        ppm_out =self.ppm(x64_2)


        # stage4中进行两次下采样，stage5不进行下采样
        return {"4": x8, "8": x16, "16": x32_4, "32": ppm_out}

    def channels(self):
        # return {"4":48,"8":128,"16":256,"32":1024}
        return {"4": 48, "8": 128, "16": 256, "32": 384}

def generate_stage2(ds,block_fun):
    blocks=[]
    for d in ds:
        blocks.append(block_fun(d))
    return blocks

class RegSegBody3wh_selfattention(nn.Module):
    def __init__(self, ds):
        super().__init__()
        gw = 16
        attention = "se"
        self.attention = PMSA_regseg(
            320,
            128,
        )
        self.stage4 = DBlock_new(32, 48, [1], gw, 2, attention)
        self.stage8 = nn.Sequential(
            DBlock_new(48, 128, [1], gw, 2, attention),
            DBlock_new(128, 128, [1], gw, 1, attention),
            DBlock_new(128, 128, [1], gw, 1, attention)
        )
        self.stage16 = nn.Sequential(
            DBlock_new(128, 256, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock_new(256, 256, d, gw, 1, attention)),
            DBlock_new(256, 320, ds[-1], gw, 1, attention)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)

        x_query = x16
        x_key = x16
        x_self_attention = self.attention(x_query,x_key)
        x16_cat = torch.cat([x_self_attention,x16],dim=1)


        return {"4": x4, "8": x8, "16": x16_cat}

    def channels(self):
        return {"4": 48, "8": 128, "16": 448}


class RegSegBody_15_selfattention(nn.Module):
    def __init__(self):
        super().__init__()
        gw = 16
        self.attention = PMSA(
            256,
            128,
        )
        self.stage4 = DBlock(32, 48, [1], gw, 2)
        self.stage8 = nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256, 256, [2], gw, 1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [2], gw, 1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)

        x_query = x32
        x_key = x32
        x_self_attention = self.attention(x_query, x_key)
        x32_cat = torch.cat([x_self_attention, x32], dim=1)

        return {"4": x4, "8": x8, "16": x16, "32": x32_cat}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}

class RegSegBody_17_wh36(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    #在stage5出来后的特征图添加transformer block
    def __init__(self,
                 d_model=1024,
                 nhead=16,
                 dim_feedforward=2048,
                 dropout=0.1
                 ):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.transformer_block = TransformerBlock(d_model,nhead,dim_feedforward,dropout)

        self.conv1 = ConvBnAct(1536, 256, 1)

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
        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        print("***********x32's first  shape***********", x32_cat.shape)
        bsz, c, h, w = x32_cat.size()
        x32 = x32_cat.view(bsz,c,h*w)
        print("***********x32's second  shape***********", x32.shape)
        x32 = x32.permute(0,2,1).contiguous()
        print("***********x32's third  shape***********", x32.shape)
        #x32 = x32.reshape(*x32.shape[:2],-1)
        #print("***********x32's second  shape***********", x32.shape)
        #x32 = x32.permute(0,2,1).contiguous()
        #print("***********x32's shape***********", x32.shape)
        x32 = self.transformer_block(x32)
        print("***********x32's forth  shape***********", x32.shape)
        print("***********transformer is ok***********")
        #x32=x32.view(8,1024,16,32)
        x32 = x32.reshape(bsz,-1,*x32_cat.shape[2:])
        print("***********x32's fifth  shape***********", x32.shape)
        #x32 = x32.permute(0,1,2).contiguous()

        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(F.relu(self.linear1(src)))
        src2 = self.dropout(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class TransformerBlock_2(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src

class TransformerBlock_3(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src

class TransformerBlock_1_channel(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(F.relu(self.linear1(src)))
        src2 = self.dropout(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class TransformerBlock_2_channel(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src

class RegSegBody_17_wh37(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    #在stage5出来后的特征图添加transformer block_2(第二种结构)
    def __init__(self,
                 d_model=1024,
                 nhead=16,
                 dim_feedforward=1024,
                 dropout=0.1
                 ):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.transformer_block_2_channel = TransformerBlock_2_channel(d_model,nhead,dim_feedforward,dropout)

        self.conv1 = ConvBnAct(1536, 256, 1)

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
        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        #print("***********x32's first  shape***********", x32_cat.shape)
        bsz, c, h, w = x32_cat.size()
        x32 = x32_cat.view(bsz,c,h*w)
        #print("***********x32's second  shape***********", x32.shape)
        x32 = x32.permute(0,2,1).contiguous()
        #print("***********x32's third  shape***********", x32.shape)
        #x32 = x32.reshape(*x32.shape[:2],-1)
        #print("***********x32's second  shape***********", x32.shape)
        #x32 = x32.permute(0,2,1).contiguous()
        #print("***********x32's shape***********", x32.shape)
        x32 = self.transformer_block_2_channel(x32)
        #print("***********x32's forth  shape***********", x32.shape)
        #print("***********transformer is ok***********")
        #x32=x32.view(8,1024,16,32)
        x32 = x32.reshape(bsz,-1,*x32_cat.shape[2:])
        #print("***********x32's fifth  shape***********", x32.shape)
        #x32 = x32.permute(0,1,2).contiguous()

        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh38(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    #在stage5出来后的特征图添加transformer block,对dim_feedforward通道数做消融实验
    def __init__(self,
                 d_model=1024,
                 nhead=16,
                 dim_feedforward=1024,
                 dropout=0.1
                 ):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.transformer_block_1_channel = TransformerBlock_1_channel(d_model,nhead,dim_feedforward,dropout)

        self.conv1 = ConvBnAct(1536, 256, 1)

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
        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32_cat = torch.cat((x32_cat, x32_7), dim=1)
        #print("***********x32's first  shape***********", x32_cat.shape)
        bsz, c, h, w = x32_cat.size()
        x32 = x32_cat.view(bsz,c,h*w)
        #print("***********x32's second  shape***********", x32.shape)
        x32 = x32.permute(0,2,1).contiguous()
        #print("***********x32's third  shape***********", x32.shape)
        #x32 = x32.reshape(*x32.shape[:2],-1)
        #print("***********x32's second  shape***********", x32.shape)
        #x32 = x32.permute(0,2,1).contiguous()
        #print("***********x32's shape***********", x32.shape)
        x32 = self.transformer_block_1_channel(x32)
        #print("***********x32's forth  shape***********", x32.shape)
        #print("***********transformer is ok***********")
        #x32=x32.view(8,1024,16,32)
        x32 = x32.reshape(bsz,-1,*x32_cat.shape[2:])
        #print("***********x32's fifth  shape***********", x32.shape)
        #x32 = x32.permute(0,1,2).contiguous()

        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh39(nn.Module): #将最后期阶段的block换为使用空洞率的block,
    # 修改block数量，减少block数量，（3，3，4）
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

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
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


    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32_1=self.stage32_1(x16)
        x32_2=self.stage32_2(x32_1)
        x32_3=self.stage32_3(x32_2)
        x32_4=self.stage32_4(x32_3)

        x_cat=torch.cat((x32_2,x32_4),dim=1)

        return {"4":x4,"8":x8,"16":x16,"32":x_cat}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":512}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh40(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    #将stride=1 的block设置为瓶颈状的
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
            DDWblock4(128, 128, dilations1, 1, 1, attention),
            DDWblock4(128, 128, dilations1, 1, 1, attention),


            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock4(256, 256, dilations1, 1, 1, attention),
            DDWblock4(256, 256, dilations1, 1, 1, attention),
        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock4(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock4(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock4(256, 256, dilations2, 1, 1, attention)
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
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh41(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    #对stride=1的block的残差连接进行消融实验
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
            DDWblock5(64, 128, dilations1, 1, 2, attention),
            DDWblock5(128, 128, dilations1, 1, 1, attention),
            DDWblock5(128, 128, dilations1, 1, 1, attention),
            DDWblock5(128, 128, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock5(128, 256, dilations1, 1, 2, attention),
            DDWblock5(256, 256, dilations1, 1, 1, attention),
            DDWblock5(256, 256, dilations1, 1, 1, attention),
            DDWblock5(256, 256, dilations1, 1, 1, attention),
            DDWblock5(256, 256, dilations1, 1, 1, attention),
            DDWblock5(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock5(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock5(256, 256, dilations2, 1, 1, attention)
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

class RegSegBody_17_wh42(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
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
            DDWblock6(128, 128, dilations1, 1, 1, attention),
            DDWblock6(128, 128, dilations1, 1, 1, attention),
            DDWblock6(128, 128, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations1, 1, 2, attention),
            DDWblock6(256, 256, dilations1, 1, 1, attention),
            DDWblock6(256, 256, dilations1, 1, 1, attention),
            DDWblock6(256, 256, dilations1, 1, 1, attention),
            DDWblock6(256, 256, dilations1, 1, 1, attention),
            DDWblock6(256, 256, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock6(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock6(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock6(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock6(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock6(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock6(256, 256, dilations2, 1, 1, attention)
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

class RegSegBody_17_wh43(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
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
            DDWblock7(128, 128, dilations1, 1, 2, attention),
            DDWblock7(128, 128, dilations1, 1, 1, attention),
            DDWblock7(128, 128, dilations1, 1, 1, attention),
            DDWblock7(128, 128, dilations1, 1, 1, attention),
            DDWblock7(128, 128, dilations1, 1, 1, attention),
            DDWblock7(128, 128, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock7(128, 128, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock7(128, 128, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock7(128, 128, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock7(128, 128, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock7(128, 128, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock7(128, 128, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock7(128, 128, dilations2, 1, 1, attention)
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