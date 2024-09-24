from torch import nn
import torch
from torch.nn import functional as F
from block_operate import AlignedModule

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
'''
class decoder_RDDNet_add(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,256,1)
        self.x8 = ConvBnAct(128,256,1)
        self.x16 = ConvBnAct(256,256,1)
        self.x32 = ConvBnAct(256,256,1)
        self.conv4 = ConvBnAct(256,256,3,1,1,groups=256)
        self.conv8 = ConvBnAct(256,256,3,1,1,groups=256)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
'''
class decoder_RDDNet_add(nn.Module):
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(256, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_stage5cat(nn.Module):  #stage5经过cat后，stage5的输出通道变为256*4=1024
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1280, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_wh(nn.Module):#将卷积修改为深度可分离卷积，存在问题，最后卷积的输出通道不为19
    def __init__(self):
        super().__init__()
        self.head32 =ConvBnAct(256, 128, 1)
        self.head16 =ConvBnAct(256, 128, 1)
        self.head8 =ConvBnAct(128, 128, 1)
        self.head4 =ConvBnAct(64, 128, 1)

        self.conv4 = nn.Sequential(
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,dilation=dilation, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )


    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.head4(x4)
        x8 = self.head8(x8)
        x16 = self.head16(x16)
        x32 = self.head32(x32)

        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False )
        x32 = x8 + x32
        #final_out.append(x32)

        x32 = self.conv8(x32)

        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x16 = x16 + x4
        #final_out.append(x16)

        x16 = self.conv4(x16)
        x16 = torch.cat((x16, x32), dim=1)
        return x16

class decoder_RDDNet_wh2(nn.Module):  #修改为深度可分离卷积
    def __init__(self):
        super().__init__()
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.head32 =ConvBnAct(256, 128, 1)
        self.head16 =ConvBnAct(256, 128, 1)
        self.head8 =ConvBnAct(128, 128, 1)
        self.head4 =ConvBnAct(64, 128, 1)

        self.conv4 = nn.Sequential(
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=out_channels,dilation=dilation, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )


    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.head4(x4)
        x8 = self.head8(x8)
        x16 = self.head16(x16)
        x32 = self.head32(x32)

        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False )
        x32 = x8 + x32
        #final_out.append(x32)

        x32 = self.conv8(x32)

        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x16 = x16 + x4
        #final_out.append(x16)

        x16 = self.conv4(x16)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
        x16 = torch.cat((x16, x32), dim=1)

        #x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)

        x16 = self.classer(self.convlast(x16))

        return x16

class decoder_RDDNet_stage5cat_C384(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(384, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_stage5cat_C1024(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_stage5cat_C1152(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1152, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_stage5cat_C1024_notup(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 2, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x8 = x4 + x8
        x4 = self.classer(self.convlast(x8))
        return x4

class decoder_RDDNet_stage5cat_C1536_channel(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(48, 128, 1)
        self.x8 = ConvBnAct(120, 128, 1)
        self.x16 = ConvBnAct(384, 128, 1)
        self.x32 = ConvBnAct(1536, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_stage5cat_C2048(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(2048, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_stage5cat_C512(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(128, 128, 1)
        self.x32 = ConvBnAct(512, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_stage4_5_cat_C1024(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(768, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_stage5cat_decoderfinal_attention(nn.Module):  #stage5经过cat后，stage5的输出通道变为256*4=1024
    #在decoder阶段，在最后两个分支融合之前，加入self-attention,因分辨率过大，导致报显存
    def __init__(self):
        super().__init__()
        self.attention = PMSA_cat(
            1024,
            128,
        )
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_query = x8
        x_key = x4
        x_self_attention = self.attention(x_query,x_key)
        x_self_attention_cat = torch.cat([x_self_attention,x4])
        x_final = self.classer(self.convlast(x_self_attention_cat))
        return x4

class decoder_RDDNet_stage5cat_decoderx32x8_attention(nn.Module):  #stage5经过cat后，stage5的输出通道变为256*4=1024
    #在decoder阶段，在最后两个分支融合之前，加入self-attention,因分辨率过大，导致报显存
    def __init__(self):
        super().__init__()
        self.attention = PMSA_cat(
            1024,
            128,
        )
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_query = x8
        x_key = x32
        x_self_attention = self.attention(x_query, x_key)
        x_self_attention = F.interpolate(x_self_attention, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #print("X-self-attention's shape", x_self_attention.shape)
        x4 = torch.cat((x4,x_self_attention),dim=1)

        x_final = self.classer(self.convlast(x4))
        return x_final

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
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.key_project = ConvBnAct(
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.value_project = ConvBnAct(
            in_channels=128,
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

class decoder_RDDNet_stage5cat_C1024_AFM(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module
    # UAFM
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.uafm_spatten=UAFM_SpAtten(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_final1 = self.uafm_spatten(x4,x16)
        print("x_final1'shape",x_final1.shape)
        x_final2 = self.uafm_spatten(x8,x32)
        print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1  #UA

class UAFM(nn.Module):
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = nn.Sequential(
            nn.Conv2d(x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias=False),
            nn.BatchNorm2d(y_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(y_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4  #查看特征图维数
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        #x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, size=x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        #self.check(x, y)
        x, y = self.prepare(x, y)   #高分辨率经过3*3卷积，低分辨率进行上采样
        out = self.fuse(x, y)
        return out

class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self._scale = nn.Parameter(torch.ones(1))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        '''atten = torch.cat([x, y], dim=1).mean(dim=1, keepdim=True)   #进行concate操作
        #print("atten'shape",atten.shape)
        atten_max = torch.cat([x, y], dim=1).max(dim=1, keepdim=True)[0]
        #print("atten_max'shape", atten.shape)
        atten = torch.cat([atten, atten_max], dim=1)'''


        atten_x_mean = x.mean(dim=1, keepdim=True)
        atten_y_mean = y.mean(dim=1, keepdim=True)

        atten_x_max = x.max(dim=1,keepdim=True)[0]
        atten_y_max = y.max(dim=1,keepdim=True)[0]
        
        atten_mean = torch.cat((atten_x_mean,atten_y_mean),dim=1)
        atten_max = torch.cat((atten_x_max,atten_y_max),dim=1)
        
        atten = torch.cat((atten_mean,atten_max),dim=1) 

        atten = self.conv_xy_atten(atten)

        out = x * atten + y * (self._scale - atten)
        out = self.conv_out(out)
        return out

class UAFM_SpAtten_S(UAFM):
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def fuse(self, x, y):
        atten = torch.cat([x, y], dim=1).mean(dim=1, keepdim=True)
        atten = self.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out

class UAFM_SpAtten_S_a(UAFM):
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()

    def fuse(self, x, y):
        atten = torch.cat([x, y], dim=1).mean(dim=1, keepdim=True)
        atten = self.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y
        out = self.conv_out(out)
        return out

class UAFM_ChAtten(UAFM):
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            nn.Conv2d(2 * y_ch, y_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(y_ch // 2, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch)
        )
        self.sigmoid = nn.Sigmoid()

    def fuse(self, x, y):
        atten = torch.cat([x, y], dim=1).mean(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
        atten = self.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out

class UAFM_ChAtten_S(UAFM):
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            nn.Conv2d(2 * y_ch, y_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(y_ch // 2, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch)
        )
        self.sigmoid = nn.Sigmoid()

    def fuse(self, x, y):
        atten = torch.cat([x, y], dim=1).mean(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
        atten = self.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out

class UAFM_ChAtten_S_a(UAFM): #α与1-α的消融实验
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            nn.Conv2d(2 * y_ch, y_ch // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(y_ch // 2, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch)
        )
        self.sigmoid = nn.Sigmoid()

    def fuse(self, x, y):
        atten = torch.cat([x, y], dim=1).mean(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
        atten = self.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y
        out = self.conv_out(out)
        return out


class AFM_Atten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self._scale = nn.Parameter(torch.ones(1))
        self.conv_high = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=False)
        self.conv_low = nn.Conv2d(128, 128, kernel_size=3,padding=1,bias=False)
        self.conv_1 = nn.Conv2d(256, 128, kernel_size=1,bias=False)
        self.act = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_2 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()


    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        '''atten = torch.cat([x, y], dim=1).mean(dim=1, keepdim=True)   #进行concate操作
        #print("atten'shape",atten.shape)
        atten_max = torch.cat([x, y], dim=1).max(dim=1, keepdim=True)[0]
        #print("atten_max'shape", atten.shape)
        atten = torch.cat([atten, atten_max], dim=1)'''

        x_up = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False)
        x_conv_low = self.conv_low(x)
        y_conv_high = self.conv_high(y)
        cat_xy = torch.cat((x_up,y_conv_high),dim=1)
        final = self.conv_1(cat_xy)
        final = self.act(final)
        final = self.global_pool(final)
        final = self.conv_2(final)
        final = self.sigmoid(final)

        out_1 = x_conv_low * final
        out_2 = y_conv_high * (self._scale - final)
        out_1 = F.interpolate(out_1, size=out_2.shape[-2:], mode='bilinear', align_corners=False)

        out = out_2 + out_1
        out = self.conv_out(out)
        return out

class AFM_SE_Atten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self._scale = nn.Parameter(torch.ones(1))
        self.conv_high = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=False)
        self.conv_low = nn.Conv2d(128, 256, kernel_size=3,padding=1,bias=False)
        self.conv_1 = nn.Conv2d(256, 128, kernel_size=1,bias=False)
        self.act = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_2 = nn.Conv2d(128,256,kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()


    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        '''atten = torch.cat([x, y], dim=1).mean(dim=1, keepdim=True)   #进行concate操作
        #print("atten'shape",atten.shape)
        atten_max = torch.cat([x, y], dim=1).max(dim=1, keepdim=True)[0]
        #print("atten_max'shape", atten.shape)
        atten = torch.cat([atten, atten_max], dim=1)'''

        #x_up = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False)
        x_conv_low = self.conv_low(x)
        y_conv_high = self.conv_high(y)
        cat_xy = torch.cat((x,y),dim=1)

        final = self.global_pool(cat_xy)
        final = self.conv_1(final)
        final = self.act(final)

        final = self.conv_2(final)
        final = self.sigmoid(final)

        final = cat_xy + final

        out_1 = x_conv_low * final
        out_2 = y_conv_high * (self._scale - final)
        out_1 = F.interpolate(out_1, size=out_2.shape[-2:], mode='bilinear', align_corners=False)

        out = out_2 + out_1
        out = self.conv_out(out)
        return out

class AttentionFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionFusionModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

        self.init_weight()

    def forward(self, feat16, feat32):
        feat32_up = F.interpolate(feat32, feat16.size()[2:], mode='nearest')
        fcat = torch.cat([feat16, feat32_up], dim=1)
        feat = self.conv(fcat)

        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return atten

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class decoder_RDDNet_stage5cat_C1024_AFM_channelattention(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module,将空间注意力修改为通道注意力
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.uafm_chatten=UAFM_ChAtten(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_final1 = self.uafm_chatten(x4,x16)
        print("x_final1'shape",x_final1.shape)
        x_final2 = self.uafm_chatten(x8,x32)
        print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1

class decoder_RDDNet_stage5cat_C1024_AFM_channelattention_S(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module,将空间注意力修改为通道注意力
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.uafm_chatten_s=UAFM_ChAtten_S(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_final1 = self.uafm_chatten_s(x4,x16)
        print("x_final1'shape",x_final1.shape)
        x_final2 = self.uafm_chatten_s(x8,x32)
        print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1

class decoder_RDDNet_stage5cat_C1024_AFM_spattention_S(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.uafm_spatten_s=UAFM_SpAtten_S(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_final1 = self.uafm_spatten_s(x4,x16)
        #print("x_final1'shape",x_final1.shape)
        x_final2 = self.uafm_spatten_s(x8,x32)
        #print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1

class decoder_RDDNet_stage5cat_C1024_AFM_channelattention_S_signal(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module,将空间注意力修改为通道注意力
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.uafm_chatten_s=UAFM_ChAtten_S(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        #x_final1 = self.uafm_chatten_s(x4,x16)
        x_final1 = self.conv4(torch.add(x4,x16))
        #print("x_final1'shape",x_final1.shape)
        x_final2 = self.uafm_chatten_s(x8,x32)
        #print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1

class decoder_RDDNet_stage5cat_C1024_AFM_spatialattention_S_a(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module,将空间注意力修改为通道注意力
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.uafm_spatten_s_a=UAFM_SpAtten_S_a(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_final1 = self.conv4(torch.add(x4,x16))
        print("x_final1'shape",x_final1.shape)
        x_final2 = self.uafm_spatten_s_a(x8,x32)
        print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1

class decoder_RDDNet_stage5cat_C1024_AFM_attanet(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module
    # AFM
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.afm_atten=AFM_Atten(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)

        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_final1 = self.afm_atten(x16,x4)
        #print("x_final1'shape",x_final1.shape)
        x_final2 = self.afm_atten(x32,x8)
        #print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1  #UA

class decoder_RDDNet_stage5cat_C1024_AFM_attanet_stage45(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module
    # AFM 在stage4、5上添加AFM，目的：验证是否分辨率相差过大导致实验指标不好
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.afm_atten=AFM_Atten(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)

        x_final1 = self.afm_atten(x32,x16)

        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final2 = self.conv4(torch.add(x4,x8))

        x_final = x_final2 + F.interpolate(x_final1, size=x_final2.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))

        return x_final

class decoder_RDDNet_stage5cat_C1024_AFM_SE_attanet(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module
    # AFM 将AFM内部修改为SE 通道注意力
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)

        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.afm_se_atten=AFM_SE_Atten(x_ch=128,y_ch=256,out_ch=256)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)

        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_final1 = self.afm_se_atten(x4,x16)
        #print("x_final1'shape",x_final1.shape)
        x_final2 = self.afm_se_atten(x8,x32)
        #print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1  #UA

class decoder_RDDNet_stage5cat_C1024_camvid(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,12,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_stage5cat_C1024_AFM_channelattention_S_signal_ADDs2s4(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module,将空间注意力修改为通道注意力
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.uafm_chatten_s=UAFM_ChAtten_S(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)

        #x_final1 = torch.add(x4,x16)
        x_final1 = self.conv4(torch.add(x4,x16))
        #print("x_final1'shape",x_final1.shape)
        x_final2 = self.uafm_chatten_s(x8,x32)
        #print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1

class decoder_RDDNet_stage5cat_C1024_AFM_channelattention_S_signal_camvid(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    #添加attention fusion module,将空间注意力修改为通道注意力
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,12,1)

        self.uafm_chatten_s=UAFM_ChAtten_S(x_ch=128,y_ch=128,out_ch=128)

    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        #x_final1 = self.uafm_chatten_s(x4,x16)
        x_final1 = self.conv4(torch.add(x4,x16))
        #print("x_final1'shape",x_final1.shape)
        x_final2 = self.uafm_chatten_s(x8,x32)
        #print("x_final2'shape",x_final2.shape)
        x_final1 = x_final1+ F.interpolate(x_final2, size=x_final1.shape[-2:], mode='bilinear', align_corners=False)
        x_final1 = self.classer(self.convlast(x_final1))
        return x_final1














class decoder_RDDNet_psanet(nn.Module):  #stage5经过cat后，stage5的输出通道变为256*4=1024
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)

        self.x8 = ConvBnAct(128, 128, 1)

        self.x32 = ConvBnAct(384, 256, 1)

        self.x_cat = ConvBnAct(384, 256, 1)

        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]

        x8 = self.x8(x8)

        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)

        x_cat = torch.cat((x32,x8),dim=1)
        x_cat =self.x_cat(x_cat)
        x_cat = F.interpolate(x_cat, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x_cat))
        return x4

class decoder_RDDNet_stage5cat_regseg(nn.Module):  #修改为regseg的decoder，自下而上
    def __init__(self):
        super().__init__()
        # self.x4 = ConvBnAct(48,128,1)
        # self.x8 = ConvBnAct(128,128,1)
        # self.x16 = ConvBnAct(256,128,1)
        # self.x32 = ConvBnAct(256,128,1)
        # self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        # self.convlast = ConvBnAct(128,32,3,1,1)
        # self.classer = nn.Conv2d(32,19,1)
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(256, 128, 1)
        self.conv16 = ConvBnAct(128, 128, 3, 1, 1, groups=1)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=1)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)

        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
        x16 = x32 + x16
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:],mode='bilinear',align_corners=False)
        x4 = torch.cat((x4,x8),dim=1)

        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_addcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = torch.cat((x4, F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)),dim=1)
        x4 = self.classer(self.convlast(x4))
        return x4
class decoder_RDDNet_catadd(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(256,256,3,1,1,groups=256)
        self.conv8 = ConvBnAct(256,256,3,1,1,groups=256)
        self.convlast = ConvBnAct(256,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.cat((x4,x16),dim=1))
        x8 = self.conv8(torch.cat((x8,x32),dim=1))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
        
class decoder_RDDNet_add3x3(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,64,3,1,1)
        self.classer = nn.Conv2d(64,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
        
class decoder_RDDNet_add64(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,64,1)
        self.x8 = ConvBnAct(128,64,1)
        self.x16 = ConvBnAct(256,64,1)
        self.x32 = ConvBnAct(256,64,1)
        self.conv4 = ConvBnAct(64,64,3,1,1,groups=64)
        self.conv8 = ConvBnAct(64,64,3,1,1,groups=64)
        self.convlast = ConvBnAct(64,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
        
class decoder_RDDNet_add32(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,32,1)
        self.x8 = ConvBnAct(128,32,1)
        self.x16 = ConvBnAct(256,32,1)
        self.x32 = ConvBnAct(256,32,1)
        self.conv4 = ConvBnAct(32,32,3,1,1,groups=32)
        self.conv8 = ConvBnAct(32,32,3,1,1,groups=32)
        self.convlast = ConvBnAct(32,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
        
class decoder_RDDNet_addcommon(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=1)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=1)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_addcommon_selfattention(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(384,128,1)
        self.conv4 = ConvBnAct(128,128,3,1,1,groups=1)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=1)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
        
class decoder_RDDNet_addcommon1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,1,groups=1)
        self.conv8 = ConvBnAct(128,128,1,groups=1)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4
        
class decoder_RDDNet_addFAM(nn.Module): 
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(48,128,1)
        self.x8 = ConvBnAct(128,128,1)
        self.x16 = ConvBnAct(256,128,1)
        self.x32 = ConvBnAct(256,128,1)
        self.conv4 = ConvBnAct(128,128,1,groups=1)
        self.conv8 = ConvBnAct(128,128,1,groups=1)
        self.convlast = ConvBnAct(128,64,1)
        self.classer = nn.Conv2d(64,19,1)
        self.FAM1 = AlignedModule(inplane=128, outplane=128)
        self.FAM2 = AlignedModule(inplane=128, outplane=128)
        self.FAM3 = AlignedModule(inplane=128, outplane=128)
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x16 = self.FAM1([x4,x16])
        x32 = self.FAM2([x8,x32])
        x4 = self.conv4(torch.add(x4,x16))
        x8 = self.conv8(torch.add(x8,x32))
        x4 = x4+self.FAM3([x4,x8])
        x4 = self.classer(self.convlast(x4))
        return x4

