from torch import nn
import torch
from torch import nn as nn
from torch.nn import functional as F
#import SelfAttentionBlock

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

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations,group_width, stride):
        super().__init__()
        self.stride = stride
        avg_downsample=True
        groups=out_channels//group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        dilation=dilations[0]
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=(1,1),groups=out_channels, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x = self.act1(x)
        if self.stride == 2:
            x=self.avg(x)
        x=self.conv2(x)+x
        x=self.bn2(x)
        x=self.act2(x)
        x=self.conv3(x)
        x=self.bn3(x)
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

def generate_stage(num,block_fun):
    blocks=[]
    for _ in range(num):
        blocks.append(block_fun())
    return blocks

class DDWblock(nn.Module): #wh
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention="se"): #inchanel=128, outchannel=128
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width  #group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation=dilations

        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)#1x1的卷积 128,128
        self.bn1=norm2d(out_channels)#BN
        self.act1=activation()#ReLU
        self.conv2=nn.Sequential(
            nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,padding=dilation,groups=out_channels,dilation=dilation,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
            #self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2=norm2d(out_channels)#BN
        self.act2=activation()#RELU
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)#1x1卷积
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

class DDWblock2(nn.Module): #wh
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention="se"): #inchanel=128, outchannel=128
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width  #group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation=6

        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)#1x1的卷积 128,128
        self.bn1=norm2d(out_channels)#BN
        self.act1=activation()#ReLU
        self.conv2=nn.Sequential(
            nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,padding=dilation,groups=out_channels,dilation=dilation,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
            #self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2=norm2d(out_channels)#BN
        self.act2=activation()#RELU
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)#1x1卷积
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

class DDWblock3(nn.Module): #wh  去掉初始1*1和最终的1*1卷积
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention="se"): #inchanel=128, outchannel=128
        super().__init__()
        avg_downsample=True
        groups=out_channels//group_width  #group_width为每个组的通道数，groups为分组个数 128//16=8

        dilation=dilations

        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3,stride=stride,padding=dilation,groups=in_channels,dilation=dilation,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
        )
        # if len(dilations)==1:
        #     dilation=dilations[0]
        #     self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        # else:#空洞卷积len(dilations)==2
        #     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, groups=group_width,
        #                            padding=2, dilation=2, bias=False)
            #self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2=norm2d(out_channels)#BN
        self.act2=activation()#RELU
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)#1x1卷积
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
            self.shortcut=Shortcut2(in_channels,out_channels,stride,avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        #x = self.act3(x + shortcut)
        x = x + shortcut
        return x



class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""
    def __init__(self, w_in, w_se):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1=nn.Conv2d(w_in, w_se, 1, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(w_se, w_in, 1, bias=True)
        self.act2=nn.Sigmoid()

    def forward(self, x):
        y=self.avg_pool(x)
        y=self.act1(self.conv1(y))
        y=self.act2(self.conv2(y))
        return x * y

class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg=nn.AvgPool2d(2,2,ceil_mode=True)
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.avg=None
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.avg is not None:
            x=self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class Shortcut2(nn.Module): #去掉avg pooling 后的1*1卷积操作
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut2, self).__init__()
        if avg_downsample and stride != 1:
            self.avg=nn.AvgPool2d(2,2,ceil_mode=True)
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.avg=None
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.avg is not None:
            x=self.avg(x)
        x= self.conv(x)
        x = self.bn(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self,w,dilations,group_width,stride,bias):
        super().__init__()
        num_splits=len(dilations)
        assert(w%num_splits==0)
        temp=w//num_splits
        assert(temp%group_width==0)
        groups=temp//group_width
        convs=[]
        for d in dilations:
            convs.append(nn.Conv2d(temp,temp,3,padding=d,dilation=d,stride=stride,bias=bias,groups=groups))
        self.convs=nn.ModuleList(convs)
        self.num_splits=num_splits
    def forward(self,x):
        x=torch.tensor_split(x,self.num_splits,dim=1)
        res=[]
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res,dim=1)

class ConvBnActConv(nn.Module):
    def __init__(self,w,stride,dilation,groups,bias):
        super().__init__()
        self.conv=ConvBnAct(w,w,3,stride,dilation,dilation,groups)
        self.project=nn.Conv2d(w,w,1,bias=bias)
    def forward(self,x):
        x=self.conv(x)
        x=self.project(x)
        return x


class YBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation,group_width, stride):
        super(YBlock, self).__init__()
        groups = out_channels // group_width
        self.conv1=nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1=norm2d(out_channels)
        self.act1=activation()
        self.conv2=nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        self.bn2=norm2d(out_channels)
        self.act2=activation()
        self.conv3=nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3=norm2d(out_channels)
        self.act3=activation()
        self.se=SEModule(out_channels,in_channels//4)
        if stride != 1 or in_channels != out_channels:
            self.shortcut=Shortcut(in_channels,out_channels,stride)
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
        x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x

class DilaBlock(nn.Module):
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

class eleven_Decoder0(nn.Module):#Sum+Cat
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

class eleven_Decoder1(nn.Module):#Sum+Sum
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 64, 1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64,64,3,1,1)
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
        x4= x8+x4
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class eleven_Decoder2(nn.Module):#Cat+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16=channels["4"],channels["8"],channels["16"]
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv8=ConvBnAct(256,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x4, x8, x16=x["4"], x["8"],x["16"]
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= torch.cat((x8,x16),dim=1)
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class LRASPP(nn.Module):
    def __init__(self, num_classes, channels, inter_channels=128):
        super().__init__()
        channels8, channels16 = channels["8"], channels["16"]
        self.cbr = ConvBnAct(channels16, inter_channels, 1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels16, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(channels8, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, x):
        # intput_shape=x.shape[-2:]
        x8, x16 = x["8"], x["16"]
        x = self.cbr(x16)
        s = self.scale(x16)
        x = x * s
        x = F.interpolate(x, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x = self.low_classifier(x8) + self.high_classifier(x)
        return x
class down32_Decoder0(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
def generate_stage2(ds,block_fun):
    blocks=[]
    for d in ds:
        blocks.append(block_fun(d))
    return blocks

class LightSeg_Encoder_0(nn.Module): #ICANet CL #################################
    def __init__(self):
        super().__init__()
        self.stem = inputlayer_0()
        self.stage4=CDBlock_0(32, 48, [1], 2)
        self.stage8=nn.Sequential(
            CDBlock_0(48, 128, [1],2),
            CDBlock_0(128, 128, [1],1)
        )
        self.stage16=nn.Sequential(
            CDBlock_0(128, 256, [1],2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4],1)
        )
        self.stage32 = nn.Sequential(
            CDBlock_0(256, 256, [1], 2),
            CDBlock_0(256,256,[2],1),
            CDBlock_0(256, 256, [4], 1),
            CDBlock_0(256, 256, [8], 1)
        )
    def forward(self,x):
        x = self.stem(x)
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=16
        attention="se"
        self.stage4=DBlock(32, 48, [1], gw, 2, attention)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2, attention),
            DBlock(128, 128, [1], gw, 1, attention),
            DBlock(128, 128, [1], gw, 1, attention)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1, attention)),
            DBlock(256, 256, ds[-1], gw, 1, attention)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}
class RegSegBody2(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1)),
            DBlock(256, 256, ds[-1], gw, 1)
        )
        self.stage32 = DBlock(256, 256, [16], gw, 2)
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody2wh(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=24
        attention="se"
        self.stage4=nn.Sequential(
            DBlock_new(32, 48, [1], gw, 2, attention),
            DBlock_new(48, 48, [1], gw, 1, attention),
        )
        self.stage8=nn.Sequential(
            DBlock_new(48, 120, [1], gw, 2, attention),
            *generate_stage(5,lambda: DBlock_new(120, 120, [1], gw, 1, attention)),
        )
        self.stage16=nn.Sequential(
            DBlock_new(120, 336, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock_new(336, 336, d, gw, 1, attention)),
            DBlock_new(336, 384, ds[-1], gw, 1, attention)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":120,"16":384}

class RegSegBodywh1(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=24
        attention="se"
        self.stage4=nn.Sequential(
            DBlock_new(32, 48, [1], gw, 2, attention),
            DBlock_new(48, 48, [1], gw, 1, attention),
        )
        self.stage8=nn.Sequential(
            DBlock_new(48, 120, [1], gw, 2, attention),
            *generate_stage(5,lambda: DBlock_new(120, 120, [1], gw, 1, attention)),
        )
        self.stage16=nn.Sequential(
            DBlock_new(120, 336, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock_new(336, 336, d, gw, 1, attention)),
            DBlock_new(336, 384, ds[-1], gw, 1, attention)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":120,"16":384}

class RegSegBody3wh(nn.Module):
    def __init__(self,ds):
        super().__init__()
        gw=16
        attention="se"
        self.stage4=DBlock_new(32, 48, [1], gw, 2, attention)
        self.stage8=nn.Sequential(
            DBlock_new(48, 128, [1], gw, 2, attention),
            DBlock_new(128, 128, [1], gw, 1, attention),
            DBlock_new(128, 128, [1], gw, 1, attention)
        )
        self.stage16=nn.Sequential(
            DBlock_new(128, 256, [1], gw, 2, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock_new(256, 256, d, gw, 1, attention)),
            DBlock_new(256, 320, ds[-1], gw, 1, attention)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)

        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":320}

class WhTest(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_1(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_2(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_3(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [4], gw, 2),
            DBlock(256, 256, [8], gw, 1),
            DBlock(256, 256, [12], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class RegSegBody_4(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class RegSegBody_5(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [6], gw, 2),
            DBlock(256, 256, [9], gw, 1),
            DBlock(256, 256, [12], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class RegSegBody_6(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_7(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_8(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_9(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )
        self.stage32=nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_10(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )
        self.stage32=nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256, 256, [6], gw, 1),
            DBlock(256, 256, [9], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class RegSegBody_11(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1),
            DBlock(48, 48, [1], gw, 1)
            )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256, 256, [8], gw, 1)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}

class RegSegBody_12(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1),
            DBlock(48, 48, [1], gw, 1)
            )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[9],gw,1),
            DBlock(256, 256, [9], gw, 1)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        return {"4":x4,"8":x8,"16":x16}
    def channels(self):
        return {"4":48,"8":128,"16":256}
        
class RegSegBody_13(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1),
            DBlock(48, 48, [1], gw, 1)
            )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8], gw, 1)
        )
        self.stage32=nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8], gw, 1)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
    
    
class RegSegBody_14(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48, 48, [1], gw, 1),
            DBlock(48, 48, [1], gw, 1)
            )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256, 256, [9], gw, 1)
        )
        self.stage32=nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[3],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256,256,[6],gw,1),
            DBlock(256, 256, [9], gw, 1)
        )
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
    
class RegSegBody_test(nn.Module):
    def __init__(self):
        super().__init__()
        gw = 16
        self.stage4 = DBlock(32, 48, [1], gw, 2)
        self.stage8 = nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256, 256, [1], gw, 1)
        )
        self.stage32_10 = DBlock(256, 128, [2], gw, 2)
        self.stage32_11 = DBlock(256, 128, [4], gw, 1)
        self.stage32_12 = DBlock(256, 128, [8], gw, 1)

        self.stage32_20 = DBlock(256, 128, [1], gw, 2)
        self.stage32_21 = DBlock(256, 128, [1], gw, 1)
        self.stage32_22 = DBlock(256, 128, [1], gw, 1)

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = torch.cat((self.stage32_20(x16) , self.stage32_10(x16)),dim=1)
        x32 = torch.cat((self.stage32_21(x32) , self.stage32_11(x32)),dim=1)
        x32 = torch.cat((self.stage32_22(x32) , self.stage32_12(x32)),dim=1)
        return {"4": x4, "16": x16, "32":x32}

    def channels(self):
        return {"4": 48, "16": 256,"32":256}
        
class down32_test(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels32=channels["4"],channels["8"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        #self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        #self.conv16 = ConvBnAct(128,64,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
        self.gloable_pool = nn.AdaptiveAvgPool2d((1,1));
    def forward(self, x):
        x4, x8,x32=x["4"], x["8"],x["32"]
        x32=self.head32(x32)
        #x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        #x16 = x16+x32
        #x16 = self.conv16(x16)
        #x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x32
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4

class RegSegBody_15(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}



class RegSegBody_16(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_17_wh(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        dilations=[1,2]
        attention = 'se'
        self.stage4=nn.Sequential(
            DDWblock(32, 64, dilations, 1, 2, attention),
            DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations, 1, 2, attention),
            DDWblock(128, 128, dilations, 1, 1, attention)
            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations, 1, 2, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention)
            #DBlock(128, 256, [1], gw, 2),
            #DBlock(256,256,[2],gw,1),
            #DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock(256, 256, dilations, 1, 2, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention)
            #DBlock(256, 256, [1], gw, 2),
            #DBlock(256,256,[4],gw,1),
            #DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh2(nn.Module): #将最后期阶段的block换为使用空洞率的block
    def __init__(self):
        super().__init__()
        gw=16
        dilations=[1,2]
        attention = 'se'
        self.stage4=nn.Sequential(
            DDWblock(32, 64, dilations, 1, 2, attention),
            DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations, 1, 2, attention),
            DDWblock(128, 128, dilations, 1, 1, attention)
            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations, 1, 2, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention)
            #DBlock(128, 256, [1], gw, 2),
            #DBlock(256,256,[2],gw,1),
            #DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock2(256, 256, dilations, 1, 2, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention)
            #DBlock(256, 256, [1], gw, 2),
            #DBlock(256,256,[4],gw,1),
            #DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh3(nn.Module): #将最后期阶段的block换为使用空洞率的block
    def __init__(self):
        super().__init__()
        gw=16
        dilations=[1,2]
        attention = 'se'
        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations, 1, 2, attention),
            DDWblock(128, 128, dilations, 1, 1, attention),
            DDWblock(128, 128, dilations, 1, 1, attention),

        #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations, 1, 2, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention)
            #DBlock(128, 256, [1], gw, 2),
            #DBlock(256,256,[2],gw,1),
            #DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock2(256, 256, dilations, 1, 2, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            #DBlock(256, 256, [1], gw, 2),
            #DBlock(256,256,[4],gw,1),
            #DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh4(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    def __init__(self):
        super().__init__()
        gw=16
        dilations=1
        attention = 'se'
        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations, 1, 2, attention),
            DDWblock(128, 128, dilations, 1, 1, attention),
            DDWblock(128, 128, dilations, 1, 1, attention),
            DDWblock(128, 128, dilations, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations, 1, 2, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention)

            #DBlock(128, 256, [1], gw, 2),
            #DBlock(256,256,[2],gw,1),
            #DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock2(256, 256, dilations, 1, 2, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention)

            #DBlock(256, 256, [1], gw, 2),
            #DBlock(256,256,[4],gw,1),
            #DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh5_d4(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
     #将空洞率设置为了4
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=4
        dilations2=1
        attention = 'se'
        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations2, 1, 2, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention)

            #DBlock(128, 256, [1], gw, 2),
            #DBlock(256,256,[2],gw,1),
            #DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention)

            #DBlock(256, 256, [1], gw, 2),
            #DBlock(256,256,[4],gw,1),
            #DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh6_d3(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #将空洞率设置为了3
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=3
        dilations2=1
        attention = 'se'
        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations2, 1, 2, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention)

            #DBlock(128, 256, [1], gw, 2),
            #DBlock(256,256,[2],gw,1),
            #DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock2(256, 256, dilations1, 1, 2, attention),
            DDWblock2(256, 256, dilations1, 1, 1, attention),
            DDWblock2(256, 256, dilations1, 1, 1, attention),
            DDWblock2(256, 256, dilations1, 1, 1, attention),
            DDWblock2(256, 256, dilations1, 1, 1, attention),
            DDWblock2(256, 256, dilations1, 1, 1, attention),
            DDWblock2(256, 256, dilations1, 1, 1, attention),
            DDWblock2(256, 256, dilations1, 1, 1, attention)

            #DBlock(256, 256, [1], gw, 2),
            #DBlock(256,256,[4],gw,1),
            #DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh4_add(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）,在一个阶段内添加add操作
    def __init__(self):
        super().__init__()
        gw=16
        dilations=[1,2]
        attention = 'se'
        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(64, 128, dilations, 1, 2, attention),
            DDWblock(128, 128, dilations, 1, 1, attention),
            DDWblock(128, 128, dilations, 1, 1, attention),
            DDWblock(128, 128, dilations, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(128, 256, dilations, 1, 2, attention),
        )
        self.stage16_add=nn.Sequential(
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention),
            DDWblock(256, 256, dilations, 1, 1, attention)
        )
        self.stage32 = nn.Sequential(
            DDWblock2(256, 256, dilations, 1, 2, attention),
        )
        self.stage32_add = nn.Sequential(
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention),
            DDWblock2(256, 256, dilations, 1, 1, attention)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16_add=self.stage16(x8)
        x16=self.stage16_add(x16_add)+x16_add
        x32_add = self.stage32(x16)
        x32 = self.stage32_add(x32_add)+x32_add
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh7_d6(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #同时去掉block内部的1*1卷积
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock3(64, 128, dilations1, 1, 2, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock3(128, 256, dilations1, 1, 2, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention)

            #DBlock(128, 256, [1], gw, 2),
            #DBlock(256,256,[2],gw,1),
            #DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 2, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention)

            #DBlock(256, 256, [1], gw, 2),
            #DBlock(256,256,[4],gw,1),
            #DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh8_d4(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #同时去掉block内部的1*1卷积，在stage5阶段内添加前向反馈，即SS-ASPP
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
            DDWblock3(64, 128, dilations1, 1, 2, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock3(128, 256, dilations1, 1, 2, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention)

            #DBlock(128, 256, [1], gw, 2),
            #DBlock(256,256,[2],gw,1),
            #DBlock(256, 256, [4], gw, 1)
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
        self.stage32_5 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 1, attention)
        )

        # self.stage32 = nn.Sequential(
        #     DDWblock3(256, 256, dilations2, 1, 2, attention),
        #     DDWblock3(256, 256, dilations2, 1, 1, attention),
        #     DDWblock3(256, 256, dilations2, 1, 1, attention),
        #     DDWblock3(256, 256, dilations2, 1, 1, attention),
        #     DDWblock3(256, 256, dilations2, 1, 1, attention),
        #     DDWblock3(256, 256, dilations2, 1, 1, attention),
        #     DDWblock3(256, 256, dilations2, 1, 1, attention),
        #     DDWblock3(256, 256, dilations2, 1, 1, attention))

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
        x32_cat = torch.cat((x32_1,x32_2),dim=1)
        x32_cat = torch.cat((x32_cat, x32_3), dim=1)
        x32_cat = torch.cat((x32_cat, x32_4), dim=1)
        x32_cat = torch.cat((x32_cat, x32_5), dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = self.conv1(x32_cat)
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh10_d4(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #同时去掉block内部的1*1卷积  设置stage5的空洞率为4
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'
        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock3(64, 128, dilations1, 1, 2, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),
            DDWblock3(128, 128, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock3(128, 256, dilations1, 1, 2, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention),
            DDWblock3(256, 256, dilations1, 1, 1, attention)

            #DBlock(128, 256, [1], gw, 2),
            #DBlock(256,256,[2],gw,1),
            #DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock3(256, 256, dilations2, 1, 2, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention),
            DDWblock3(256, 256, dilations2, 1, 1, attention)

            #DBlock(256, 256, [1], gw, 2),
            #DBlock(256,256,[4],gw,1),
            #DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh11(nn.Module):  # 将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 将空洞率设置为了为1，2，2，4，4，6，6，7  第二种为1，2，2，3，4，6，6，8
    def __init__(self):
        super().__init__()
        gw = 16
        dilations2 = 1
        attention = 'se'
        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            # DBlock(32, 48, [1], gw, 2),
            # DBlock(48,48,[1],gw,1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations2, 1, 2, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),

            # DBlock(48, 128, [1], gw, 2),
            # DBlock(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention)

            # DBlock(128, 256, [1], gw, 2),
            # DBlock(256,256,[2],gw,1),
            # DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock(256, 256, 1, 1, 2, attention),
            DDWblock(256, 256, 1, 1, 1, attention),
            DDWblock(256, 256, 1, 1, 1, attention),
            DDWblock(256, 256, 1, 1, 1, attention),
            DDWblock(256, 256, 1, 1, 1, attention),
            DDWblock(256, 256, 1, 1, 1, attention),
            DDWblock(256, 256, 1, 1, 1, attention),

            # DBlock(256, 256, [1], gw, 2),
            # DBlock(256,256,[4],gw,1),
            # DBlock(256, 256, [8],gw,1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}


class RegSegBody_17_wh12(nn.Module):  # 将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 将空洞率设置为了4  修改block数量为 4 5 7
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 4
        dilations2 = 1
        attention = 'se'
        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            # DBlock(32, 48, [1], gw, 2),
            # DBlock(48,48,[1],gw,1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations2, 1, 2, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),

            # DBlock(48, 128, [1], gw, 2),
            # DBlock(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            #DDWblock(256, 256, dilations2, 1, 1, attention)

            # DBlock(128, 256, [1], gw, 2),
            # DBlock(256,256,[2],gw,1),
            # DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            #DDWblock(256, 256, dilations1, 1, 1, attention)

            # DBlock(256, 256, [1], gw, 2),
            # DBlock(256,256,[4],gw,1),
            # DBlock(256, 256, [8],gw,1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}

class RegSegBody_17_wh13(nn.Module):  # 将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    # 将空洞率设置为了4  修改block数量为 4 4 6
    def __init__(self):
        super().__init__()
        gw = 16
        dilations1 = 4
        dilations2 = 1
        attention = 'se'
        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            # DBlock(32, 48, [1], gw, 2),
            # DBlock(48,48,[1],gw,1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations2, 1, 2, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),

            # DBlock(48, 128, [1], gw, 2),
            # DBlock(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            #DDWblock(256, 256, dilations2, 1, 1, attention),
            #DDWblock(256, 256, dilations2, 1, 1, attention)

            # DBlock(128, 256, [1], gw, 2),
            # DBlock(256,256,[2],gw,1),
            # DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock(256, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            #DDWblock(256, 256, dilations1, 1, 1, attention),
            #DDWblock(256, 256, dilations1, 1, 1, attention)

            # DBlock(256, 256, [1], gw, 2),
            # DBlock(256,256,[4],gw,1),
            # DBlock(256, 256, [8],gw,1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}

class RegSegBody_17_wh14(nn.Module):  # 将最后期阶段的block换为使用空洞率的block,增加block数量，
    # block分布457将空洞率设置为了为1，2，3，4，4,6，8
    def __init__(self):
        super().__init__()
        gw = 16
        dilations2 = 1
        attention = 'se'
        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            # DBlock(32, 48, [1], gw, 2),
            # DBlock(48,48,[1],gw,1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations2, 1, 2, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),

            # DBlock(48, 128, [1], gw, 2),
            # DBlock(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention)

            # DBlock(128, 256, [1], gw, 2),
            # DBlock(256,256,[2],gw,1),
            # DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock(256, 256, 1, 1, 2, attention),
            DDWblock(256, 256, 2, 1, 1, attention),
            DDWblock(256, 256, 3, 1, 1, attention),
            DDWblock(256, 256, 4, 1, 1, attention),
            DDWblock(256, 256, 4, 1, 1, attention),
            DDWblock(256, 256, 6, 1, 1, attention),
            DDWblock(256, 256, 8, 1, 1, attention)

            # DBlock(256, 256, [1], gw, 2),
            # DBlock(256,256,[4],gw,1),
            # DBlock(256, 256, [8],gw,1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}

class RegSegBody_17_wh15(nn.Module):  # 将最后期阶段的block换为使用空洞率的block,增加block数量，
    # block分布457将空洞率设置为了为1，2，3，4，6,6，6
    def __init__(self):
        super().__init__()
        gw = 16
        dilations2 = 1
        attention = 'se'
        self.stage4 = nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            # DBlock(32, 48, [1], gw, 2),
            # DBlock(48,48,[1],gw,1)
        )
        self.stage8 = nn.Sequential(
            DDWblock(64, 128, dilations2, 1, 2, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),
            DDWblock(128, 128, dilations2, 1, 1, attention),

            # DBlock(48, 128, [1], gw, 2),
            # DBlock(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention),
            DDWblock(256, 256, dilations2, 1, 1, attention)

            # DBlock(128, 256, [1], gw, 2),
            # DBlock(256,256,[2],gw,1),
            # DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DDWblock(256, 256, 1, 1, 2, attention),
            DDWblock(256, 256, 2, 1, 1, attention),
            DDWblock(256, 256, 3, 1, 1, attention),
            DDWblock(256, 256, 4, 1, 1, attention),
            DDWblock(256, 256, 6, 1, 1, attention),
            DDWblock(256, 256, 6, 1, 1, attention),
            DDWblock(256, 256, 6, 1, 1, attention)

            # DBlock(256, 256, [1], gw, 2),
            # DBlock(256,256,[4],gw,1),
            # DBlock(256, 256, [8],gw,1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}

class RegSegBody_17_wh16(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将1，3，5，7后的特征图进行cat    之后再进行add
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

        x32_add = x32_1 + x32_3
        x32_add = x32_add + x32_5
        x32_add = x32_add + x32_7
        x32 =x32_add
        #x32_cat = torch.cat((x32_1,x32_3),dim=1)
        #x32_cat = torch.cat((x32_cat, x32_5), dim=1)
        #x32 = torch.cat((x32_cat, x32_7), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        #return {"4":48,"8":128,"16":256,"32":1024}
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
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

class RegSegBody_17_wh18_add(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行add    之后再进行add
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

        x32_add = x32_2 + x32_4
        x32_add = x32_add + x32_6
        x32_add = x32_add + x32_7
        x32 =x32_add
        #x32_cat = torch.cat((x32_1,x32_3),dim=1)
        #x32_cat = torch.cat((x32_cat, x32_5), dim=1)
        #x32 = torch.cat((x32_cat, x32_7), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        #return {"4":48,"8":128,"16":256,"32":1024}
        return {"4":48,"8":128,"16":256,"32":256}


class RegSegBody_17_wh22(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat  修改block数为5，6，8
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

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
        x32_8=self.stage32_8(x32_7)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = torch.cat((x32_cat, x32_8), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh24(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat  修改block数为5，6，8
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1),
            ConvBnAct(64, 64, 3, 1, 1)
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
        x32_8=self.stage32_8(x32_7)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = torch.cat((x32_cat, x32_8), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh25(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat  修改block数为5，6，8
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1),
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
        x32_8=self.stage32_8(x32_7)
        x32_9=self.stage32_9(x32_8)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = torch.cat((x32_cat, x32_8), dim=1)
        x32 = torch.cat((x32, x32_9), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1280}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh27(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat  修改block数为5，6，8
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1),
            ConvBnAct(64, 64, 3, 1, 1)
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
        x32_8=self.stage32_8(x32_7)
        x32_9=self.stage32_9(x32_8)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = torch.cat((x32_cat, x32_9), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh28(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat  修改block数为5，6，8
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.stage4=nn.Sequential(
            ConvBnAct(32, 64, 3, 2, 1),
            ConvBnAct(64, 64, 3, 1, 1),
            ConvBnAct(64, 64, 3, 1, 1)
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
        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)
        x32_8=self.stage32_8(x32_7)
        x32_9=self.stage32_9(x32_8)
        x32_10=self.stage32_10(x32_9)

        x32_cat = torch.cat((x32_2,x32_4), dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32_cat = torch.cat((x32_cat, x32_8), dim=1)
        x32 = torch.cat((x32_cat,x32_10), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":1280}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh29(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'



        self.stage4=nn.Sequential(
            ConvBnAct(32, 48, 3, 2, 1)
            # DDWblock(32, 64, dilations, 1, 2, attention),
            # DDWblock(64, 64, dilations, 1, 1, attention)
            #DBlock(32, 48, [1], gw, 2),
            #DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DDWblock(48, 120, dilations1, 1, 2, attention),
            DDWblock(120, 120, dilations1, 1, 1, attention),
            DDWblock(120, 120, dilations1, 1, 1, attention),
            DDWblock(120, 120, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DDWblock(120, 384, dilations1, 1, 2, attention),
            DDWblock(384, 384, dilations1, 1, 1, attention),
            DDWblock(384, 384, dilations1, 1, 1, attention),
            DDWblock(384, 384, dilations1, 1, 1, attention),
            DDWblock(384, 384, dilations1, 1, 1, attention),
            DDWblock(384, 384, dilations1, 1, 1, attention)

        )

        self.stage32_1 = nn.Sequential(
            DDWblock(384, 384, dilations2, 1, 2, attention)
        )
        self.stage32_2 = nn.Sequential(
            DDWblock(384, 384, dilations2, 1, 1, attention)
        )
        self.stage32_3 = nn.Sequential(
            DDWblock(384, 384, dilations2, 1, 1, attention)
        )
        self.stage32_4 = nn.Sequential(
            DDWblock(384, 384, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(384, 384, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(384, 384, dilations2, 1, 1, attention)
        )
        self.stage32_7 = nn.Sequential(
            DDWblock(384, 384, dilations2, 1, 1, attention)
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
        return {"4":48,"8":120,"16":384,"32":1536}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh31(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
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
            DDWblock(64, 256, dilations1, 1, 2, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),
            DDWblock(256, 256, dilations1, 1, 1, attention),

            #DBlock(48, 128, [1], gw, 2),
            #DBlock(128, 128, [1], gw, 1)
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
        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = torch.cat((x32_cat, x32_7), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":64,"8":256,"16":256,"32":2048}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh32(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
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
        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)

        x32_cat = torch.cat((x32_2,x32_4),dim=1)
        x32_cat = torch.cat((x32_cat, x32_6), dim=1)
        x32 = torch.cat((x32_cat, x32_7), dim=1)
        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":64,"8":128,"16":256,"32":2048}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh33(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
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

        self.stage16_1 = nn.Sequential(
            DDWblock(128, 256, dilations2, 1, 2, attention)
        )
        self.stage16_2 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage16_3 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage16_4 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage16_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage16_6 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
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

        #x16=self.stage16(x8)
        x16_1=self.stage16_1(x8)
        x16_2=self.stage16_2(x16_1)
        x16_3=self.stage16_3(x16_2)
        x16_4=self.stage16_4(x16_3)
        x16_5=self.stage16_5(x16_4)
        x16_6=self.stage16_6(x16_5)

        x16=torch.cat((x16_2,x16_4),dim=1)
        x16=torch.cat((x16,x16_6),dim=1)

        x32_1=self.stage32_1(x16_6)
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
        return {"4":48,"8":128,"16":768,"32":1024}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh34(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
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
            DDWblock(512, 256, dilations2, 1, 1, attention)
        )
        self.stage32_5 = nn.Sequential(
            DDWblock(256, 256, dilations2, 1, 1, attention)
        )
        self.stage32_6 = nn.Sequential(
            DDWblock(512, 256, dilations2, 1, 1, attention)
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
        x32_3=torch.cat((x32_2,x32_3),dim=1)

        x32_4=self.stage32_4(x32_3)
        x32_5=self.stage32_5(x32_4)
        x32_5=torch.cat((x32_4,x32_5),dim=1)

        x32_6=self.stage32_6(x32_5)
        x32_7=self.stage32_7(x32_6)
        x32_7=torch.cat((x32_6,x32_7),dim=1)


        #x32 = self.conv1(x32_cat)  #x32_cat 的通道数为256*4=1024
        #x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32_7}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":512}
        #return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17_wh17_satge2block(nn.Module): #将最后期阶段的block换为使用空洞率的block,增加block数量，（4，6，8）
    #在stage5阶段内添加前向反馈，即SS-ASPP  将2，4，6，7后的特征图进行cat
    def __init__(self):
        super().__init__()
        gw=16
        dilations1=1
        dilations2=4
        attention = 'se'

        self.conv1 = ConvBnAct(1536, 256, 1)

        self.stage4=nn.Sequential(
            #ConvBnAct(32, 64, 3, 2, 1)
            DDWblock(32, 64, dilations1, 1, 2, attention),
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









class RegSegBody_17(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16

        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_17Y(nn.Module):
    def __init__(self):
        super().__init__()
        gw=1
        self.stage4=nn.Sequential(
            YBlock(32, 48, [1], gw, 2),
            YBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            YBlock(48, 128, [1], gw, 2),
            YBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            YBlock(128, 256, [1], gw, 2),
            YBlock(256,256,[2],gw,1),
            YBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            YBlock(256, 256, [1], gw, 2),
            YBlock(256,256,[4],gw,1),
            YBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_17D(nn.Module):
    def __init__(self):
        super().__init__()
        gw=1
        self.stage4=nn.Sequential(
            DilaBlock(32, 48, [1], gw, 2),
            DilaBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DilaBlock(48, 128, [1], gw, 2),
            DilaBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DilaBlock(128, 256, [1], gw, 2),
            DilaBlock(256,256,[1,2],gw,1),
            DilaBlock(256, 256, [1,4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DilaBlock(256, 256, [1], gw, 2),
            DilaBlock(256,256,[1,4],gw,1),
            DilaBlock(256, 256, [1,8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_18(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_19(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
class RegSegBody_20(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_21(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256, 256, [2], gw, 1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
 
class RegSegBody_22(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage160=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage161 = nn.Sequential(
            DBlock(256, 256, [1], gw, 1),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x160=self.stage160(x8)
        x161 = self.stage161(x160)
        return {"4":x4,"8":x8,"16":x160,"32":x161}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_23(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1)
        )
        self.stage80=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage81=nn.Sequential(
            DBlock(128, 256, [1], gw, 1),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage82 = nn.Sequential(
            DBlock(256, 256, [1], gw, 1),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x80=self.stage80(x4)
        x81=self.stage81(x80)
        x82 = self.stage82(x81)
        return {"4":x4,"8":x80,"16":x81,"32":x82}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_24(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}



class RegSegBody_25(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 1),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 1),
            DBlock(256,256,[1],gw,1),
            DBlock(256, 256, [4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_26(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 1),
            DBlock(256,256,[1],gw,1),
            DBlock(256, 256, [4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
class RegSegBody_27(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256, 256, [4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
        
class RegSegBody_28(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_29(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[1],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[4],gw,1),
            DBlock(256,256,[8],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class RegSegBody_30(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=nn.Sequential(
            DBlock(32, 48, [1], gw, 2),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1),
            DBlock(48,48,[1],gw,1)
        )
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
            DBlock(128, 128, [1], gw, 1),
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256,256,[4],gw,1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[4],gw,1),
            DBlock(256, 256, [8],gw,1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}

class RegSegBody_addallshort(nn.Module):
    def __init__(self):
        super().__init__()
        gw = 16
        self.stage4 = DBlock_noshort(32, 48, [1], gw, 2)
        self.stage8 = nn.Sequential(
            DBlock_addallshort(48, 128, [1], gw, 2),
            DBlock_addallshort(128, 128, [1], gw, 1),
            DBlock_addallshort(128, 128, [1], gw, 1)
        )
        self.stage16 = nn.Sequential(
            DBlock_addallshort(128, 256, [1], gw, 2),
            DBlock_addallshort(256, 256, [2], gw, 1),
            DBlock_addallshort(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock_addallshort(256, 256, [1], gw, 2),
            DBlock_addallshort(256, 256, [4], gw, 1),
            DBlock_addallshort(256, 256, [8], gw, 1)
        )

    def forward(self, x):
        x4 = self.stage4(x)
        x8 = self.stage8(x4)
        x16 = self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4": x4, "8": x8, "16": x16, "32": x32}

    def channels(self):
        return {"4": 48, "8": 128, "16": 256, "32": 256}
        

class RegSegBody_152(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 320, [1], gw, 2),
            DBlock(320,320,[2],gw,1),
            DBlock(320, 320, [4], gw, 1),
            DBlock(320, 320, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":320}

class RegSegBody_153(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=DBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            DBlock(48, 128, [1], gw, 2),
            DBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            DBlock(128, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            DBlock(256, 256, [1], gw, 2),
            DBlock(256,256,[2],gw,1),
            DBlock(256, 256, [4], gw, 1),
            DBlock(256, 256, [8], gw, 1)
        )
        self.short_x4 = ConvBnAct(48,128,1,stride=2)
        self.short_x8 = ConvBnAct(128,256,1,stride=2)
    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8+self.short_x4(x4))
        x32 = self.stage32(x16+self.short_x8(x8))
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}


class down32_Decoder1(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
        self.gloable_pool = nn.AdaptiveAvgPool2d((1,1));
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class RegSegBody_repvgg(nn.Module):
    def __init__(self):
        super().__init__()
        gw=16
        self.stage4=SBlock(32, 48, [1], gw, 2)
        self.stage8=nn.Sequential(
            SBlock(48, 128, [1], gw, 2),
            SBlock(128, 128, [1], gw, 1)
        )
        self.stage16=nn.Sequential(
            SBlock(128, 256, [1], gw, 2),
            SBlock(256,256,[2],gw,1),
            SBlock(256, 256, [4], gw, 1)
        )
        self.stage32 = nn.Sequential(
            SBlock(256, 256, [1], gw, 2),
            SBlock(256,256,[2],gw,1),
            SBlock(256, 256, [4], gw, 1),
            SBlock(256, 256, [8], gw, 1)
        )

    def forward(self,x):
        x4=self.stage4(x)
        x8=self.stage8(x4)
        x16=self.stage16(x8)
        x32 = self.stage32(x16)
        return {"4":x4,"8":x8,"16":x16,"32":x32}
    def channels(self):
        return {"4":48,"8":128,"16":256,"32":256}
        
class down32_Decoder64(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 64, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+64,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoder128(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 128, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+128,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoder_cat(nn.Module):#cat+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(256,128,3,1,1)
        self.conv8=ConvBnAct(256,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = torch.cat((x32,x16),dim=1)
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= torch.cat((x16,x8),dim=1)
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoder_sum(nn.Module):#sum+sum+sum
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 64, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x32+x16
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x16+x8
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4= x8+x4
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoderprocess64(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 64, 1)
        self.head16=ConvBnAct(channels16, 64, 1)
        self.head8=ConvBnAct(channels8, 64, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(64,64,3,1,1)
        self.conv8=ConvBnAct(64,64,3,1,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoder1x1(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,1)
        self.conv8=ConvBnAct(128,64,1)
        self.conv4=ConvBnAct(64+8,64,3,1,1) 
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_Decoderban(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 24, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(64+24,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4
        
class down32_DecoderFAM(nn.Module):#Sum+Cat
    def __init__(self, num_classes, channels):
        super().__init__()
        channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(channels32, 128, 1)
        self.head16=ConvBnAct(channels16, 128, 1)
        self.head8=ConvBnAct(channels8, 128, 1)
        self.head4=ConvBnAct(channels4, 8, 1)
        self.conv16 = ConvBnAct(128,128,3,1,1)
        self.conv8=ConvBnAct(128,64,3,1,1)
        self.conv4=ConvBnAct(128+128+64+8,64,3,1,1)
        self.classifier=nn.Conv2d(64, num_classes, 1)
    def forward(self, x):
        x4, x8, x16,x32=x["4"], x["8"],x["16"],x["32"]
        x32=self.head32(x32)
        x16=self.head16(x16)
        x8=self.head8(x8)
        x4=self.head4(x4)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x16 = x16+x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)#数组上采样操作
        x8= x8 + x16
        x8=self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4=torch.cat((x32,x16,x8,x4),dim=1)
        x4=self.conv4(x4)
        x4=self.classifier(x4)
        return x4