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

class RegSegBody_17_wh17_tfblock(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
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
        x32 = torch.cat((x32_cat, x32_7), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}