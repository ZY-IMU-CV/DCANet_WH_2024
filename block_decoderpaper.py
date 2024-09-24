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

class decoder_RDDNet_C256_paper(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
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

class decoder_RDDNet_C256_paper_PPM(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(128, 128, 1)
        self.x8 = ConvBnAct(256, 128, 1)
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

class decoder_RDDNet_C384_paper(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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

class decoder_RDDNet_C640_papers6(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x4 = ConvBnAct(128, 128, 1)
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(640, 128, 1)
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

class decoder_RDDNet_C640_papers6_last4stage(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x4 = ConvBnAct(128, 128, 1)
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(640, 128, 1)
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

class decoder_RDDNet_C384_papers6(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x4 = ConvBnAct(128, 128, 1)
        self.x8 = ConvBnAct(256, 128, 1)
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

class decoder_RDDNet_C640_papers5(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x4 = ConvBnAct(128, 128, 1)
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(640, 128, 1)
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

class decoder_RDDNet_C640_papers5_camvid(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x4 = ConvBnAct(128, 128, 1)
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(640, 128, 1)
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

class decoder_RDDNet_psaa(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x32 = ConvBnAct(640, 128, 1)
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

class decoder_RDDNet_C640_papers5_normal(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x4 = ConvBnAct(128, 128, 1)
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
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

class decoder_RDDNet_C640_papers6cat(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x4 = ConvBnAct(128, 128, 1)
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(640, 128, 1)
        self.conv4 = ConvBnAct(256, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(256,128,3,1,1,groups=128)
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
        x4_cat = torch.cat((x4,x16),dim=1)
        x8_cat = torch.cat((x8,x32),dim=1)

        x4 = self.conv4(x4_cat)
        x8 = self.conv8(x8_cat)
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4

class decoder_RDDNet_C896_papers6(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x4 = ConvBnAct(128, 128, 1)
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(896, 128, 1)
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

class decoder_RDDNet_C256_paper_channel5(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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

class decoder_RDDNet_C256_paper_channel3(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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

class decoder_RDDNet_C256_paper_channel1(nn.Module):
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
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(512, 128, 1)
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

class decoder_RDDNet_C256_paper_channel2(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
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

class decoder_RDDNet_C256_paper_channel4(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x8 = ConvBnAct(256, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
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

class decoder_RDDNet_C512_paper(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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

class decoder_RDDNet_C768_paper(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x32 = ConvBnAct(768, 128, 1)
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

class decoder_RDDNet_C1024_paper(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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

class decoder_RDDNet_C1536_paper(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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

class decoder_RDDNet_C1024DW_paper(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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

        self.conv4_dw = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.conv8_dw = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
        )

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
        x4 = self.conv4_dw(torch.add(x4,x16))
        x8 = self.conv8_dw(torch.add(x32,x8))
        x4 = x4+ F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.classer(self.convlast(x4))
        return x4



class decoder_RDDNet_C1792_paper(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.x32 = ConvBnAct(1792, 128, 1)
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

        atten_x_max = x.max(dim=1, keepdim=True)[0]
        atten_y_max = y.max(dim=1, keepdim=True)[0]

        atten_mean = torch.cat((atten_x_mean, atten_y_mean), dim=1)
        atten_max = torch.cat((atten_x_max, atten_y_max), dim=1)

        atten = torch.cat((atten_mean, atten_max), dim=1)

        atten = self.conv_xy_atten(atten)

        out = x * atten + y * (self._scale - atten)
        out = self.conv_out(out)
        return out

class UAFM_ChAtten(UAFM):
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            nn.Conv2d(2 * y_ch, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(y_ch, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch)
        )
        self.sigmoid = nn.Sigmoid()

    def fuse(self, x, y):

        #atten_x_max = x.max(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)#channel=128
        #atten_y_max = y.max(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)#channel=128

        atten_x_mean = x.mean(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)#channel=128
        atten_y_mean = y.mean(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)#channel=128

        atten_mean_cat = torch.cat((atten_x_mean,atten_y_mean),dim=1)#channel=256
        #atten_max_cat = torch.cat((atten_x_max, atten_y_max), dim=1)#channel=256
        #atten = torch.cat((atten_mean_cat,atten_max_cat),dim=1)#channel=512

        atten = self.sigmoid(self.conv_xy_atten(atten_mean_cat))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out

class UAFM_ChAtten_cat(UAFM):
    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            nn.Conv2d(2 * y_ch, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(y_ch, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch)
        )
        self.sigmoid = nn.Sigmoid()

    def fuse(self, x, y):
        atten_mean = torch.cat([x, y], dim=1).mean(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)#channel=256
        #atten_max = torch.cat([x, y], dim=1).max(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)#channel=256
        #atten = torch.cat((atten_mean,atten_max),dim=1)#channel=512
        atten = self.sigmoid(self.conv_xy_atten(atten_mean))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out

class UAFM_SpAtten_cat(UAFM):
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
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
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

        atten_mean = torch.cat([x, y], dim=1).mean(dim=1, keepdim=True)
        atten_max = torch.cat([x, y], dim=1).max(dim=1, keepdim=True)[0]

        atten = torch.cat([atten_mean, atten_max], dim=1)

        atten = self.conv_xy_atten(atten)

        out = x * atten + y * (self._scale - atten)
        out = self.conv_out(out)
        return out

class UAFM_SpAtten_catmean(UAFM):
    #先进行mean池化再进行cat
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
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
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

        atten_mean = torch.cat([x, y], dim=1).mean(dim=1, keepdim=True)
        #atten_max = torch.cat([x, y], dim=1).max(dim=1, keepdim=True)[0]

        #atten = torch.cat([atten_mean, atten_max], dim=1)

        atten = self.conv_xy_atten(atten_mean)

        out = x * atten + y * (self._scale - atten)
        out = self.conv_out(out)
        return out

class UAFM_SpAtten_Channel_cat(UAFM):
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

        self.conv_xy_atten_sp = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_xy_atten_ch = nn.Sequential(
            nn.Conv2d(y_ch*2, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(y_ch, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch)
        )
        self.sigmoid = nn.Sigmoid()

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

        spatten_mean = torch.cat([x, y], dim=1).mean(dim=1, keepdim=True)
        #atten_max = torch.cat([x, y], dim=1).max(dim=1, keepdim=True)[0]
        chatten_mean = torch.cat([x, y], dim=1).mean(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
        #atten = torch.cat([atten_mean, atten_max], dim=1)
        atten_sp = self.conv_xy_atten_sp(spatten_mean)
        atten_ch = self.conv_xy_atten_ch(chatten_mean)
        out_sp = x * atten_sp + y * (self._scale - atten_sp)
        out_ch = x * atten_ch + y * (self._scale - atten_ch)
        out = out_sp + out_ch
        out = self.conv_out(out)
        return out

class MFAF(UAFM):
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

        self.conv_xy_atten_sp = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_xy_atten_ch = nn.Sequential(
            nn.Conv2d(y_ch*2, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(y_ch, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch)
        )
        self.sigmoid = nn.Sigmoid()

        self._scale = nn.Parameter(torch.ones(1))

    def fuse(self, x, y, z):
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

        spatten_mean = y.mean(dim=1, keepdim=True)
        #spatten_max = y.max(dim=1, keepdim=True)
        chatten_mean = y.mean(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
        #chatten_max = y.max(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
        spatten_cat = torch.cat([spatten_mean,spatten_mean],dim=1)
        chatten_cat = torch.cat([chatten_mean,chatten_mean],dim=1)

        spatten_cat = self.conv_xy_atten_sp(spatten_cat)
        chatten_cat = self.conv_xy_atten_ch(chatten_cat)

        out_atten = spatten_cat * chatten_cat

        x_out = x * out_atten
        z_out = z * out_atten

        out = x_out + z_out

        return out

class UAFM_SpAtten_Channel(UAFM):
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

        self.conv_xy_atten_sp = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_xy_atten_ch = nn.Sequential(
            nn.Conv2d(y_ch*2, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(y_ch, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch)
        )
        self.sigmoid = nn.Sigmoid()

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

        spatten_mean_x = x.mean(dim=1,keepdim=True)
        spatten_mean_y = y.mean(dim=1, keepdim=True)

        chatten_mean_x = x.mean(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
        chatten_mean_y = y.mean(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

        spatten_mean = torch.cat((spatten_mean_x,spatten_mean_y),dim=1)
        chatten_mean = torch.cat((chatten_mean_x,chatten_mean_y),dim=1)


        #atten = torch.cat([atten_mean, atten_max], dim=1)

        atten_sp = self.conv_xy_atten_sp(spatten_mean)

        atten_ch = self.conv_xy_atten_ch(chatten_mean)

        out_sp = x * atten_sp + y * (self._scale - atten_sp)
        out_ch = x * atten_ch + y * (self._scale - atten_ch)

        out = out_sp + out_ch

        out = self.conv_out(out)
        return out

class UAFM_SpAtten_Channel_cat_max(UAFM):
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

        self.conv_xy_atten_sp = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_xy_atten_ch = nn.Sequential(
            nn.Conv2d(y_ch*2, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(y_ch, y_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(y_ch)
        )
        self.sigmoid = nn.Sigmoid()

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


        #spatten_mean = torch.cat([x, y], dim=1).mean(dim=1, keepdim=True)
        #chatten_mean = torch.cat([x, y], dim=1).mean(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

        spatten_max = torch.cat([x, y], dim=1).max(dim=1, keepdim=True)[0]
        chatten_max = torch.cat([x, y], dim=1).max(dim=1, keepdim=True)[0]


        #atten = torch.cat([atten_mean, atten_max], dim=1)

        atten_sp = self.conv_xy_atten_sp(spatten_mean)

        atten_ch = self.conv_xy_atten_ch(chatten_mean)

        out_sp = x * atten_sp + y * (self._scale - atten_sp)
        out_ch = x * atten_ch + y * (self._scale - atten_ch)

        out = out_sp + out_ch

        out = self.conv_out(out)
        return out

class decoder_RDDNet_C1024_paper_UAFMSpatten(nn.Module):  #
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
        #空间注意力中，所传进去的参数并未使用，分别对x,y进行max与mean池化操作，cat之后变为4通道的特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMChatten(nn.Module):  #
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
        #空间注意力中，分别对x,y进行max与mean池化操作，cat之后变为512通道的特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_chatten(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMChatten_cat(nn.Module):  #stage5经过cat后，stage5的输出通道变为256+128=384
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
        self.uafm_channel_cat=UAFM_ChAtten_cat(x_ch=128,y_ch=128,out_ch=128)
        #通道注意力中，先对x与y进行cat后，再对x,y进行max与mean池化操作，
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_cat(nn.Module):  #
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
        self.uafm_spatten_cat=UAFM_SpAtten_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_catmean(nn.Module):
    #先进行mean池化，在进行cat
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
        self.uafm_spatten_catmean=UAFM_SpAtten_catmean(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_catmean(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat(nn.Module):  #
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
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_MFAF(nn.Module):  #
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
        self.conv8 = ConvBnAct(128,128,3,2,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.mfaf=MFAF(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
        x8 = self.conv8(x8)

        x_mfaf = self.mfaf(x8,x16,x32)

        x_final = F.interpolate(x_mfaf, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class DWConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(DWConvBnAct, self).__init__()
        self.avg = nn.AvgPool2d(2,2,ceil_mode=True)
        self.stride=stride
        self.conv3x3=nn.Conv2d(in_channels,in_channels,3,stride=1,padding=padding,groups=in_channels)
        self.conv1x1=nn.Conv2d(in_channels,out_channels,1)
        self.bn=norm2d(out_channels)
        if apply_act:
            self.act=activation()
        else:
            self.act=None
    def forward(self, x):
        if self.stride ==2:
            x= self.avg(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x

class decoder_RDDNet_C1024_paper_ICA(nn.Module):#Sum+Cat
    def __init__(self):
        super().__init__()
        #channels4,channels8,channels16,channels32=channels["4"],channels["8"],channels["16"],channels["32"]
        self.head32 = ConvBnAct(1024, 128, 1)
        self.head16=ConvBnAct(256, 128, 1)
        self.head8=ConvBnAct(128, 128, 1)
        self.head4=ConvBnAct(64, 8, 1)
        self.conv16 =ConvBnAct(128,128,3,1,1,groups=64)
        self.conv8=ConvBnAct(128,64,3,1,1,groups=64)
        self.conv4=DWConvBnAct(72,64,3,1,1)
        self.classifier=nn.Conv2d(64, 19, 1)
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

class decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_channel1(nn.Module):  #
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
        self.x8 = ConvBnAct(64, 128, 1)
        self.x16 = ConvBnAct(128, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_channel2(nn.Module):  #
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
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_channel4(nn.Module):  #
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
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C512_paper_UAFMSpatten_Channel_cat(nn.Module):  #
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
        self.x32 = ConvBnAct(512, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C256_paper_UAFMSpatten_Channel_cat(nn.Module):  #
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
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1536_paper_UAFMSpatten_Channel_cat(nn.Module):  #
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
        self.x32 = ConvBnAct(1536, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_Channel(nn.Module):  #
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
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_x16x32(nn.Module):  #
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        #x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm_x32 = self.uafm_spatten_channel_cat(x8,x32)
        x_uafm_x16 = self.uafm_spatten_channel_cat(x4,x16)
        x_final = x_uafm_x16 + F.interpolate(x_uafm_x32, size=x_uafm_x16.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_x16(nn.Module):  #
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        #x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        #x_uafm_x32 = self.uafm_spatten_channel_cat(x8,x32)
        x_uafm_x16 = self.uafm_spatten_channel_cat(x4,x16)
        x8 = self.conv8(torch.add(x8,x32))
        x_final = x_uafm_x16 + F.interpolate(x8, size=x_uafm_x16.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_cat_345(nn.Module):  #
    def __init__(self):
        super().__init__()
        self.x4 = ConvBnAct(64, 128, 1)
        self.x8 = ConvBnAct(128, 128, 1)
        self.x16 = ConvBnAct(256, 128, 1)
        self.x32 = ConvBnAct(1024, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        #x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm_x32 = self.uafm_spatten_channel_cat(x8,x32)
        x_uafm_x16 = self.uafm_spatten_channel_cat(x4,x16)
        #x_final = x_uafm_x16 + F.interpolate(x_uafm_x32, size=x_uafm_x16.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.uafm_spatten_channel_cat(x_uafm_x16,x_uafm_x32)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1280_paper_UAFMSpatten_Channel(nn.Module):  #
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
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C2560_paper_UAFMSpatten_Channel(nn.Module):  #
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
        self.x32 = ConvBnAct(2560, 128, 1)
        self.conv4 = ConvBnAct(128, 128, 3, 1, 1, groups=128)
        self.conv8 = ConvBnAct(128,128,3,1,1,groups=128)
        self.convlast = ConvBnAct(128,32,3,1,1)
        self.classer = nn.Conv2d(32,19,1)
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

class decoder_RDDNet_C1024_paper_UAFMSpatten_Channel_catcamvid(nn.Module):  #
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
        self.uafm_spatten_channel_cat=UAFM_SpAtten_Channel_cat(x_ch=128,y_ch=128,out_ch=128)
        #空间注意力中，所传进去的参数并未使用，先对x,y进行cat，再对cat后的特征图进行max与mean池化操作，最终生成两个特征图
    def forward(self,x):
        x4, x8, x16, x32 = x["4"], x["8"], x["16"], x["32"]
        x4 = self.x4(x4)
        x8 = self.x8(x8)
        x16 = self.x16(x16)
        x32 = self.x32(x32)
        #UAFM的prepare_y中会进行上采样
        #x32 = F.interpolate(x32, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x16 = F.interpolate(x16, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.conv4(torch.add(x4,x16))
        #x8 = self.conv8(torch.add(x32,x8))
        x_uafm = self.uafm_spatten_channel_cat(x8,x32)
        x_final = x4 + F.interpolate(x_uafm, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x_final = self.classer(self.convlast(x_final))
        return x_final

