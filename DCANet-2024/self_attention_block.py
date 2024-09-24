import torch
from mmcv.cnn import ConvModule, constant_init
from torch import nn as nn
from torch.nn import functional as F


class SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels  #2048+512=2560
        self.query_in_channels = query_in_channels # 2048
        self.out_channels = out_channels #2048
        self.channels = channels  #512
        self.share_key_query = share_key_query #False
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,  #2560
            channels,        #512
            num_convs=key_query_num_convs, # =1
            use_conv_module=key_query_norm, #false
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,  #with_out=None out_channels=2048
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)  #返回通道数？C*H*W
        query = self.query_project(query_feats) #对query特征图进行1*1卷积
        if self.query_downsample is not None:   #=None
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1) #将特征图调整为CXHW
        # 转置操作 query.shape=(1,CHANNEL,H,W),shape[0]:图像的的垂直尺寸（高度）shape[1]图像的水平尺寸（宽度）shape[2]图像的通道数
        query = query.permute(0, 2, 1).contiguous() #进行维度变换 0代表一个，1代表channel，2代表HW,变换为HWXC

        key = self.key_project(key_feats) #对经过ASPP 的key特征图进行1*1卷积
        value = self.value_project(key_feats) #对经过ASPP 的value特征图进行1*1卷积
        if self.key_downsample is not None: #=none
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)  # -1的含义为根据第一个值动态调整第二个值的大小
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key) #对key特征图和query特征图进行矩阵相乘
        if self.matmul_norm:               #matmul_norm=none
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  #对特征图的每一行进行softmax

        context = torch.matmul(sim_map, value)  #经过自注意力后的特征图与value特征图进行相乘操作
        context = context.permute(0, 2, 1).contiguous() #进行维度变换 调换回原来（1，C,HW）
        context = context.reshape(batch_size, -1, *query_feats.shape[2:]) #将（1，C,HW）调整为（1，C，H，W）
        if self.out_project is not None:
            context = self.out_project(context)
        return context
