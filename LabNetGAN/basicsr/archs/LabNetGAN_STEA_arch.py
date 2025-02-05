import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.nn.parameter import Parameter
from torch.nn import Softmax
from torch import nn
import math
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class UnetDownsample(nn.Module):
    def __init__(self, n_feat):
        super(UnetDownsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UnetUpsample(nn.Module):
    def __init__(self, n_feat):
        super(UnetUpsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
#
#
#class NonLocalSparseAttention(nn.Module):
#    def __init__(self, n_hashes=4, channels=64, k_size=3, reduction=4, chunk_size=144, conv=default_conv,
#                 res_scale=1):
#        super(NonLocalSparseAttention, self).__init__()
#        self.chunk_size = chunk_size
#        self.n_hashes = n_hashes
#        self.reduction = reduction
#        self.res_scale = res_scale
#        self.conv_match = BasicBlock(conv, channels, channels // reduction, k_size, bn=False, act=None)
#        self.conv_assembly = BasicBlock(conv, channels, channels, 1, bn=False, act=None)
#
#    def LSH(self, hash_buckets, x):
#        # x: [N,H*W,C]
#        N = x.shape[0]
#        device = x.device
#
#        # generate random rotation matrix
#        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)  # [1,C,n_hashes,hash_buckets//2]
#        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1,
#                                                                                             -1)  # [N, C, n_hashes, hash_buckets//2]
#
#        # locality sensitive hashing
#        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations)  # [N, n_hashes, H*W, hash_buckets//2]
#        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)  # [N, n_hashes, H*W, hash_buckets]
#
#        # get hash codes
#        hash_codes = torch.argmax(rotated_vecs, dim=-1)  # [N,n_hashes,H*W]
#
#        # add offsets to avoid hash codes overlapping between hash rounds
#        offsets = torch.arange(self.n_hashes, device=device)
#        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
#        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,))  # [N,n_hashes*H*W]
#
#        return hash_codes
#
#    def add_adjacent_buckets(self, x):
#        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
#        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
#        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)
#
#    def forward(self, input):
#
#        N, _, H, W = input.shape
#        x_embed = self.conv_match(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
#        y_embed = self.conv_assembly(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
#        L, C = x_embed.shape[-2:]
#
#        # number of hash buckets/hash bits
#        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)
#
#        # get assigned hash codes/bucket number
#        hash_codes = self.LSH(hash_buckets, x_embed)  # [N,n_hashes*H*W]
#        hash_codes = hash_codes.detach()
#
#        # group elements with same hash code by sorting
#        _, indices = hash_codes.sort(dim=-1)  # [N,n_hashes*H*W]
#        _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
#        mod_indices = (indices % L)  # now range from (0->H*W)
#        x_embed_sorted = batched_index_select(x_embed, mod_indices)  # [N,n_hashes*H*W,C]
#        y_embed_sorted = batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C]
#
#        # pad the embedding if it cannot be divided by chunk_size
#        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
#        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes, -1, C))  # [N, n_hashes, H*W,C]
#        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes, -1, C * self.reduction))
#        if padding:
#            pad_x = x_att_buckets[:, :, -padding:, :].clone()
#            pad_y = y_att_buckets[:, :, -padding:, :].clone()
#            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
#            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)
#
#        x_att_buckets = torch.reshape(x_att_buckets, (
#        N, self.n_hashes, -1, self.chunk_size, C))  # [N, n_hashes, num_chunks, chunk_size, C]
#        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C * self.reduction))
#
#        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)
#
#        # allow attend to adjacent buckets
#        x_match = self.add_adjacent_buckets(x_match)
#        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)
#
#        # unormalized attention score
#        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets,
#                                 x_match)  # [N, n_hashes, num_chunks, chunk_size, chunk_size*3]
#
#        # softmax
#        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
#        score = torch.exp(raw_score - bucket_score)  # (after softmax)
#        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])
#
#        # attention
#        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets)  # [N, n_hashes, num_chunks, chunk_size, C]
#        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))
#
#        # if padded, then remove extra elements
#        if padding:
#            ret = ret[:, :, :-padding, :].clone()
#            bucket_score = bucket_score[:, :, :-padding].clone()
#
#        # recover the original order
#        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
#        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N,n_hashes*H*W]
#        ret = batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
#        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]
#
#        # weighted sum multi-round attention
#        ret = torch.reshape(ret, (N, self.n_hashes, L, C * self.reduction))  # [N, n_hashes*H*W,C]
#        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
#        probs = nn.functional.softmax(bucket_score, dim=1)
#        ret = torch.sum(ret * probs, dim=1)
#
#        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input
#        return ret


#class NonLocalAttention(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, softmax_scale=10, average=True, res_scale=1,conv=default_conv):
#        super(NonLocalAttention, self).__init__()
#        self.res_scale = res_scale
#        self.conv_match1 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
#        self.conv_match2 = BasicBlock(conv, channel, channel//reduction, 1, bn=False, act = nn.PReLU())
#        self.conv_assembly = BasicBlock(conv, channel, channel, 1,bn=False, act=nn.PReLU())
#        
#    def forward(self, input):
#        x_embed_1 = self.conv_match1(input)
#        x_embed_2 = self.conv_match2(input)
#        x_assembly = self.conv_assembly(input)
#
#        N,C,H,W = x_embed_1.shape
#        x_embed_1 = x_embed_1.permute(0,2,3,1).view((N,H*W,C))
#        x_embed_2 = x_embed_2.view(N,C,H*W)
#        score = torch.matmul(x_embed_1, x_embed_2)
#        score = F.softmax(score, dim=2)
#        x_assembly = x_assembly.view(N,-1,H*W).permute(0,2,1)
#        x_final = torch.matmul(score, x_assembly)
#        return x_final.permute(0,2,1).view(N,-1,H,W)+self.res_scale*input



#class attention2d(nn.Module):
#    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
#        super(attention2d, self).__init__()
#        assert temperature%3==1
#        self.avgpool = nn.AdaptiveAvgPool2d(1)
#        if in_planes!=3:
#            hidden_planes = int(in_planes*ratios)+1
#        else:
#            hidden_planes = K
#        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
#        # self.bn = nn.BatchNorm2d(hidden_planes)
#        self.gelu = nn.GELU()
#        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
#        self.temperature = temperature
#        if init_weight:
#            self._initialize_weights()
#
#
#    def _initialize_weights(self):
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                if m.bias is not None:
#                    nn.init.constant_(m.bias, 0)
#            if isinstance(m ,nn.BatchNorm2d):
#                nn.init.constant_(m.weight, 1)
#                nn.init.constant_(m.bias, 0)
#
#    def updata_temperature(self):
#        if self.temperature!=1:
#            self.temperature -=3
#            print('Change temperature to:', str(self.temperature))
#
#
#    def forward(self, x):
#        x = self.avgpool(x)
#        x = self.fc1(x)
#        x = self.gelu(x)
#        x = self.fc2(x).view(x.size(0), -1)
#        return F.softmax(x/self.temperature, 1)
#
#
#class Dynamic_conv2d(nn.Module):
#    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
#        super(Dynamic_conv2d, self).__init__()
#        assert in_planes%groups==0
#        self.in_planes = in_planes
#        self.out_planes = out_planes
#        self.kernel_size = kernel_size
#        self.stride = stride
#        self.padding = padding
#        self.dilation = dilation
#        self.groups = groups
#        self.bias = bias
#        self.K = K
#        self.attention = attention2d(in_planes, ratio, K, temperature)
#
#        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
#        if bias:
#            self.bias = nn.Parameter(torch.zeros(K, out_planes))
#        else:
#            self.bias = None
#        if init_weight:
#            self._initialize_weights()
#
#        #TODO 初始化
#    def _initialize_weights(self):
#        for i in range(self.K):
#            nn.init.kaiming_uniform_(self.weight[i])
#
#
#    def update_temperature(self):
#        self.attention.updata_temperature()
#
#    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
#        softmax_attention = self.attention(x)
#        batch_size, in_planes, height, width = x.size()
#        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
#        weight = self.weight.view(self.K, -1)
#
#        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
#        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
#        if self.bias is not None:
#            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
#            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
#                              dilation=self.dilation, groups=self.groups*batch_size)
#        else:
#            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
#                              dilation=self.dilation, groups=self.groups * batch_size)
#
#        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
#        return output



#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#
#        self.conv1x1_wv = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_b = nn.Conv2d(channel, 1, 1)
#
#        self.conv1x1_w1 = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_w2 = nn.Conv2d(channel, channel, 1)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        v = self.conv1x1_wv(x).view((N,H*W,C))
#
#        w1 = self.conv1x1_w1(x)
#
#        b = self.conv1x1_b(x)
#
#        w2 = self.conv1x1_w2(x).permute(0,2,3,1).view((N,H*W,C))
#
#        out = w1 * b
#        out = out.view((N,C,H*W))
#        out = torch.matmul(out, w2)
#        
#        out = torch.matmul(v, out)
#        out = out.view((N,H,W,C)).permute(0,3,1,2)
#        
#        return out * self.beta + x


#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#
#        self.conv1x1_wv = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_b = nn.Conv2d(channel, 1, 1)
#
#        self.w1 = nn.Sequential(
#            nn.Conv2d(channel, channel, 1),
#            Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=7, ratio=0.25, padding=7//2, groups=channel),
#            Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=9, ratio=0.25, padding=((9//2)*4), groups=channel, dilation=4)
#        )
#
#        self.w2 = nn.Sequential(
#            nn.Conv2d(channel, channel, 1),
#            Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=7, ratio=0.25, padding=7//2, groups=channel),
#            Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=9, ratio=0.25, padding=((9//2)*4), groups=channel, dilation=4)
#        )
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        v = self.conv1x1_wv(x).view((N,H*W,C))
#
#        w1 = self.w1(x)
#
#        b = self.conv1x1_b(x)
#
#        w2 = self.w2(x).permute(0,2,3,1).view((N,H*W,C))
#
#        out = w1 * b
#        out = out.view((N,C,H*W))
#        out = torch.matmul(out, w2)
#        
#        out = torch.matmul(v, out)
#        out = out.view((N,H,W,C)).permute(0,3,1,2)
#        
#        return out * self.beta + x



#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#
#        self.conv1x1_wv = nn.Conv2d(channel, channel, 1)
#
#        self.b = nn.Sequential(
#            nn.Conv2d(channel, 1, 1),
#            nn.Conv2d(1, 1, 7, padding=7//2, groups=1),
#            nn.Conv2d(1, 1, 9, stride=1, padding=((9//2)*4), groups=1, dilation=4)
#        )
#
#        self.conv1x1_w1 = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_w2 = nn.Conv2d(channel, channel, 1)
#
#        self.conv7x7_last = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=7, ratio=0.25, padding=7//2, groups=channel)
#        self.conv1x1_last = nn.Conv2d(channel, channel, 1)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        v = self.conv1x1_wv(x).view((N,H*W,C))
#
#        w1 = self.conv1x1_w1(x)
#
#        b = self.b(x)
#
#        w2 = self.conv1x1_w2(x).permute(0,2,3,1).view((N,H*W,C))
#
#        out = w1 * b
#        out = out.view((N,C,H*W))
#        out = torch.matmul(out, w2)
#        
#        out = torch.matmul(v, out)
#        out = out.view((N,H,W,C)).permute(0,3,1,2)
#
#        out = self.conv7x7_last(out)
#        out = self.conv1x1_last(out)
#        
#        return out * self.beta + x


#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#
#        self.conv1x1_b = nn.Conv2d(channel, 1, 1)
#        self.conv7x7_b = nn.Conv2d(1, 1, 7, padding=7//2, groups=1)
#        self.conv_spatial_b = nn.Conv2d(1, 1, 9, stride=1, padding=((9//2)*4), groups=1, dilation=4)
#
#        self.conv1x1_w1 = nn.Conv2d(channel, channel, 1)
#        self.conv7x7_w1 = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=7, ratio=0.25, padding=7//2, groups=channel)
#        self.conv_spatial_w1 = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=9, ratio=0.25, padding=((9//2)*4), groups=channel, dilation=4)
#
#        self.conv1x1_w2 = nn.Conv2d(channel, channel, 1)
#        self.conv7x7_w2 = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=7, ratio=0.25, padding=7//2, groups=channel)
#        self.conv_spatial_w2 = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=9, ratio=0.25, padding=((9//2)*4), groups=channel, dilation=4)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        conv_w1 = self.conv1x1_w1(x)
#        conv_w1 = self.conv7x7_w1(conv_w1)
#        conv_w1 = self.conv_spatial_w1(conv_w1)
#
#        conv_b = self.conv1x1_b(x)
#        conv_b = self.conv7x7_b(conv_b)
#        conv_b = self.conv_spatial_b(conv_b)
#
#        conv_w2 = self.conv1x1_w2(x)
#        conv_w2 = self.conv7x7_w2(conv_w2)
#        conv_w2 = self.conv_spatial_w2(conv_w2)
#        
#        out = conv_w1 * x
#        out = out * conv_b
#        out = out * conv_w2
#        
#        return out * self.beta + x


#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=64, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#
#        self.conv1x1_wv = nn.Conv2d(channel, channel, 1)
#
#        self.b = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_w1 = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_w2 = nn.Conv2d(channel, channel, 1)
#
#        self.last = nn.Conv2d(channel, channel, 1)
#
##        self.conv7x7_last = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=7, ratio=0.25, padding=7//2, groups=channel)
##        self.conv1x1_last = nn.Conv2d(channel, channel, 1)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        v = self.conv1x1_wv(x).view((N,C,H*W))
#
#        w1 = self.conv1x1_w1(x)
#
#        b = self.b(x)
#
#        w2 = self.conv1x1_w2(x).permute(0,2,3,1).view((N,H*W,C))
#
#        out = w1 * b
#        out = out.view((N,C,H*W))
#        out = torch.matmul(out, w2)
#
#        score = F.softmax(out, dim=2)
#        
#        out = torch.matmul(score, v)
#        out = out.view((N,C,H,W))
#
#        out = self.last(out)
#
##        out = self.conv7x7_last(out)
##        out = self.conv1x1_last(out)
#        
#        return out + x


#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=64, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#
#        self.wv = nn.Sequential(
#            nn.Conv2d(channel, channel, kernel_size=1),
#            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=7//2, groups=channel)
#        )
#
#        self.b = nn.Sequential(
#            nn.Conv2d(channel, 1, 1),
#            nn.Conv2d(1, 1, 7, padding=7//2, groups=1),
#            nn.Conv2d(1, 1, 9, stride=1, padding=((9//2)*4), groups=1, dilation=4)
#        )
#
#        self.w1 = nn.Sequential(
#            nn.Conv2d(channel, channel, kernel_size=1),
#            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=7//2, groups=channel)
#        )
#
#        self.w2 = nn.Sequential(
#            nn.Conv2d(channel, channel, kernel_size=1),
#            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=7//2, groups=channel)
#        )
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        v = self.wv(x).view((N,C,H*W))
#
#        w1 = self.w1(x)
#
#        b = self.b(x)
#
#        w2 = self.w2(x).permute(0,2,3,1).view((N,H*W,C))
#
#        out = w1 * b
#        out = out.view((N,C,H*W))
#        out = torch.matmul(out, w2)
#
#        score = F.softmax(out, dim=2)
#        
#        out = torch.matmul(score, v)
#        out = out.view((N,C,H,W))
#        
#        return out * self.beta + x


#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#        self.conv_match1 = BasicBlock(conv, channel, channel, 1, bn=False, act=None)   
#        
#        self.conv7x7_last = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=7, ratio=0.25, padding=7//2, groups=channel)
#        self.conv1x1_last = nn.Conv2d(channel, channel, 1)
#
#        self.b = nn.Sequential(
#            nn.Conv2d(channel, channel, 7, padding=7//2, groups=channel),
#            nn.Conv2d(channel, channel, 9, stride=1, padding=((9//2)*4), groups=channel, dilation=4),
#            nn.Conv2d(channel, channel, 1)
#        )
#
#        self.k = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        b = self.b(x)
#        
#        x_embed_1 = self.conv_match1(x)
#        out = b * x_embed_1
#        
#        out = self.conv7x7_last(self.k * out)
#        out = self.conv1x1_last(out)
#        
#        return out + x


#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=64, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#
#        self.conv1 = nn.Conv2d(channel, channel*4, kernel_size=1)
#        self.dwconv3 = nn.Conv2d(channel*4, channel*4, kernel_size=7, stride=1, padding=7//2, groups=channel*4)
#        self.last = nn.Conv2d(channel, channel, kernel_size=1)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#        all = self.dwconv3(self.conv1(x))
#        w1,b,w2,wv = all.chunk(4, dim=1)
#        out = w1 * b
#        out = out.view((N,C,H*W))
#        out = torch.matmul(out, w2.view((N,H*W,C)))
#        score = F.softmax(out, dim=-1)
#        out = torch.matmul(score, wv.view((N,C,H*W)))
#        out = out.view((N,C,H,W))
#        out = self.last(out)
#        return out + x


#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#
#        self.conv1x1_wv = nn.Conv2d(channel, channel, 1)
#
#        self.b = nn.Sequential(
#            nn.Conv2d(channel, 1, 1),
#            nn.Conv2d(1, 1, 7, padding=7//2, groups=1),
#            nn.Conv2d(1, 1, 9, stride=1, padding=((9//2)*4), groups=1, dilation=4)
#        )
#
#        self.conv1x1_w1 = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_w2 = nn.Conv2d(channel, channel, 1)
#
#        self.conv7x7_last = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=7, ratio=0.25, padding=7//2, groups=channel)
#        self.conv1x1_last = nn.Conv2d(channel, channel, 1)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        v = self.conv1x1_wv(x).view((N,H*W,C))
#
#        w1 = self.conv1x1_w1(x)
#
#        b = self.b(x)
#
#        w2 = self.conv1x1_w2(x).permute(0,2,3,1).view((N,H*W,C))
#
#        out = w1 * b
#        out = out.view((N,C,H*W))
#        out = torch.matmul(out, w2)
#        
#        out = torch.matmul(v, out)
#        out = out.view((N,H,W,C)).permute(0,3,1,2)
#
#        out = self.conv7x7_last(out)
#        out = self.conv1x1_last(out)
#        
#        return out * self.beta + x

#
#class DynamicVANBlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(DynamicVANBlock, self).__init__()
#
#        self.conv1x1_wv = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_b = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_w1 = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1_w2 = nn.Conv2d(channel, channel, 1)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        v = self.conv1x1_wv(x)
#
#        w1 = self.conv1x1_w1(x)
#
#        b = self.conv1x1_b(x)
#
#        w2 = self.conv1x1_w2(x)
#
#        out = w1 * b
#        out = out * w2
#        
#        out = v * out
#        
#        return out * self.beta + x


#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(ENLABlock, self).__init__()
#
#        self.w1 = nn.Sequential(
#            nn.Conv2d(channel, channel, kernel_size=1),
#            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=7//2, groups=channel),
#            nn.Conv2d(channel, channel, kernel_size=1)
#        )
#
#        self.b = nn.Conv2d(channel, channel, 1)
#
#        self.w2 = nn.Sequential(
#            nn.Conv2d(channel, channel, kernel_size=1),
#            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=7//2, groups=channel),
#            nn.Conv2d(channel, channel, kernel_size=1)
#        )
#
#        self.wv = nn.Conv2d(channel, channel, 1)
#
#        self.last = nn.Conv2d(channel, channel, 1)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#        
#        w1 = self.w1(x)
#
#        b = self.b(x)
#
#        w2 = self.w2(x)
#
#        v = self.wv(x)
#
#        out1 = w1 * b
#        out2 = w2 * v
#        out = out1 * out2
#        out = self.last(out)
#        
#        return out + x

#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(ENLABlock, self).__init__()
#
#        self.w1 = nn.Sequential(
#            nn.Conv2d(channel, channel, 1),
#            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=7//2, groups=channel)
#        )
#
#        self.b = nn.Sequential(
#            nn.Conv2d(channel, channel, 1)
#        )
#
#        self.mid_conv = nn.Conv2d(channel, channel, 1)
#
#        self.w2 = nn.Sequential(
#            nn.Conv2d(channel, channel, 1),
#            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=3//2, groups=channel)
#        )
#
#        self.wv = nn.Sequential(
#            nn.Conv2d(channel, channel, 1),
#            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=3//2, groups=channel)
#        )
#
#        self.last = nn.Conv2d(channel, channel, 1)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#        
#        w1 = self.w1(x)
#
#        b = self.b(x)
#
#        w2 = self.w2(x).view((N,H*W,C))
#
#        v = self.wv(x)
#
#        out = self.mid_conv(w1 * b)
#        out = torch.matmul(out.view((N,C,H*W)), w2)
#        score = F.softmax(out, dim=-1)
#        out = torch.matmul(score, v.view((N,C,H*W)))
#        out = out.view((N,C,H,W))
#        out = self.last(out)
#        
#        return out + x


#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(ENLABlock, self).__init__()
#        self.conv_match1 = BasicBlock(conv, channel, channel, 1, bn=False, act=None)   
#        
#        self.conv7x7_last = Dynamic_conv2d(in_planes=channel, out_planes=channel, kernel_size=7, ratio=0.25, padding=7//2, groups=channel)
#        self.conv1x1_last = nn.Conv2d(channel, channel, 1)
#
#        self.conv1x1 = nn.Conv2d(channel, 1, 1)
#        self.conv7x7 = nn.Conv2d(1, 1, 7, padding=7//2, groups=1)
#        self.conv_spatial = nn.Conv2d(1, 1, 9, stride=1, padding=((9//2)*4), groups=1, dilation=4)
#        
##        self.conv7x7 = Dynamic_conv2d(in_planes=1, out_planes=1, kernel_size=7, ratio=0.25, padding=7//2, groups=1)
##        self.conv_spatial = Dynamic_conv2d(in_planes=1, out_planes=1, kernel_size=9, ratio=0.25, padding=((9//2)*4), groups=1, dilation=4)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#
#        conv_eig = self.conv1x1(x)
#        conv_eig = self.conv7x7(conv_eig)
#        conv_eig = self.conv_spatial(conv_eig)
#        
#        x_embed_1 = self.conv_match1(x)
#        out = conv_eig * x_embed_1
#        
#        out = self.conv7x7_last(self.beta * out)
#        out = self.conv1x1_last(out)
#        
#        return out + x


#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(ENLABlock, self).__init__()
#
#        self.w1 = nn.Conv2d(channel, channel, 1)
#
#        self.eig = nn.Conv2d(1, channel, channel)
#
#        self.wv = nn.Parameter(torch.zeros((1, 1, channel, channel)), requires_grad=True)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
##        print(isinstance(x, torch.FloatTensor))
#
#        x_proj = self.w1(x).view((N,C,H*W))
#
#        x_T = x.permute(0,2,3,1).view((N,H*W,C))
#
#        A = torch.matmul(x.view((N,C,H*W)), x_T).unsqueeze(0)
#
##        print(A.shape)
#
#        A_eig = self.eig(A)
#
##        print(A_eig.shape)
#
#        B = torch.diag_embed(A_eig.permute(0,2,3,1).squeeze(0).squeeze(0))
##        print(A_eig.permute(0,2,3,1).squeeze(0).squeeze(0))
##        print(B.shape)
#        with torch.no_grad():
#            L, V = torch.linalg.eig(A.squeeze(0))
#            V_inv = torch.linalg.inv(V)
##        print(L.shape)
##        print(V.shape)
#
#        A_new = V.real @ B @ V_inv.real
#
#        A_proj = self.wv * A_new
#
#        out = torch.matmul(A_proj, x_proj).view((N,C,H,W))
#        
#        return out * self.beta + x


#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(ENLABlock, self).__init__()
#
#        self.w1 = nn.Conv2d(channel, channel, 1)
#
##        self.eig = nn.Conv2d(1, channel, channel)
#
#        self.wv = nn.Parameter(torch.zeros((1, 1, channel, channel)), requires_grad=True)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
##        print(isinstance(x, torch.FloatTensor))
#
#        x_proj = self.w1(x).view((N,C,H*W))
#
#        x_T = x.permute(0,2,3,1).view((N,H*W,C))
#
#        A = torch.matmul(x.view((N,C,H*W)), x_T).unsqueeze(0)
#
##        print(A.shape)
#
##        A_eig = self.eig(A)
#
##        print(A_eig.shape)
#
##        B = A_eig.permute(0,2,3,1).squeeze(0).squeeze(0)
##        print(A_eig.permute(0,2,3,1).squeeze(0).squeeze(0))
##        print(B.shape)
#        with torch.no_grad():
#            L, V = torch.linalg.eig(A.squeeze(0).detach())
#            V_inv = torch.linalg.inv(V.detach())
##        print(L.shape)
##        print(V.shape)
#
#        A_temp = V.data.real.detach() @ torch.diag_embed(L.real.detach())
#        A_new = torch.matmul(A_temp, V_inv.data.real.detach())
#
#        A_proj = self.wv * A_new
#
#        out = torch.matmul(A_proj, x_proj).view((N,C,H,W))
#        
#        return out * self.beta + x


#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(ENLABlock, self).__init__()
#
#        self.w1 = nn.Conv2d(channel, channel, 1)
#
#        self.mlp = nn.Sequential(
#            nn.Linear(channel, channel // 4),
#            nn.GELU(),
#            nn.Linear(channel // 4, channel)
#        )
#
#        self.wv = nn.Parameter(torch.zeros((1, 1, channel, channel)), requires_grad=True)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
##        print(isinstance(x, torch.FloatTensor))
#
#        x_proj = self.w1(x).view((N,C,H*W))
#
#        x_T = x.permute(0,2,3,1).view((N,H*W,C))
#
#        A = torch.matmul(x.view((N,C,H*W)), x_T).unsqueeze(0)
#
##        print(A.shape)
#
##        B = torch.diag_embed(A_eig.permute(0,2,3,1).squeeze(0).squeeze(0))
##        print(A_eig.permute(0,2,3,1).squeeze(0).squeeze(0))
##        print(B.shape)
#        with torch.no_grad():
#            L, V = torch.linalg.eig(A.squeeze(0))
#            V_inv = torch.linalg.inv(V)
##            print(L.shape)
##            print(V.shape)
#
#        L_proj = self.mlp(L.real.unsqueeze(0))
#        A_new = V.real @ torch.diag_embed(L_proj.squeeze(0)) @ V_inv.real
#
#        A_proj = self.wv * A_new
#
#        out = torch.matmul(A_proj, x_proj).view((N,C,H,W))
#        
#        return out * self.beta + x



#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(ENLABlock, self).__init__()
#
#        self.w1 = nn.Conv2d(channel, channel, 1)
#
##        self.mlp = nn.Sequential(
##            nn.Linear(channel, channel // 4),
##            nn.GELU(),
##            nn.Linear(channel // 4, channel)
##        )
#
#        self.proj = nn.Sequential(
#            nn.Conv2d(channel, channel // 4, 1),
#            nn.GELU(),
#            nn.Conv2d(channel // 4, channel, 1)
#        )
#
#        self.wv = nn.Parameter(torch.zeros((1, 1, channel, channel)), requires_grad=True)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
##        print(isinstance(x, torch.FloatTensor))
#
#        x_proj = self.w1(x).view((N,C,H*W))
#
#        x_T = x.permute(0,2,3,1).view((N,H*W,C))
#
#        A = torch.matmul(x.view((N,C,H*W)), x_T).unsqueeze(0)
#
##        print(A.shape)
#        L, V = torch.linalg.eig(A.squeeze(0))
#        V_inv = torch.linalg.inv(V)
##        print(L.shape)
##        print(V.shape)
#
#        L_proj = self.proj(L.real.unsqueeze(0).unsqueeze(0).permute(0,3,1,2).contiguous())
#        A_new = V.real @ torch.diag_embed(L_proj.permute(0,2,3,1).squeeze(0).squeeze(0).contiguous()) @ V_inv.real
#
#        A_proj = self.wv * A_new
#
#        out = torch.matmul(A_proj, x_proj).view((N,C,H,W))
#        
#        return out * self.beta + x


#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(ENLABlock, self).__init__()
#
#        self.w1 = nn.Conv2d(channel, channel, 1)
#
##        self.mlp = nn.Sequential(
##            nn.Linear(channel, channel // 4),
##            nn.GELU(),
##            nn.Linear(channel // 4, channel)
##        )
#
#        self.proj = nn.Sequential(
#            nn.Conv2d(channel, channel // 4, 1),
#            nn.GELU(),
#            nn.Conv2d(channel // 4, channel, 1)
#        )
#
#        self.wv = nn.Parameter(torch.zeros((1, 1, channel, channel)), requires_grad=True)
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
##        print(isinstance(x, torch.FloatTensor))
#
#        x_proj = self.w1(x).view((N,C,H*W))
#
#        x_T = x.permute(0,2,3,1).view((N,H*W,C))
#
#        A = torch.matmul(x.view((N,C,H*W)), x_T).unsqueeze(0)
#
##        print(A.shape)
#        L, V = torch.linalg.eig(A.squeeze(0))
#        V_inv = torch.linalg.inv(V)
##        print(L.shape)
##        print(V.shape)
#
#        L_proj = self.proj(L.real.unsqueeze(0).unsqueeze(0).permute(0,3,1,2).contiguous())
#        A_new = V.real @ torch.diag_embed(L_proj.permute(0,2,3,1).squeeze(0).squeeze(0).contiguous()) @ V_inv.real
#
#        A_proj = self.wv * A_new
#
#        out = torch.matmul(A_proj, x_proj).view((N,C,H,W))
#        
#        return out * self.beta + x


#class FitSoftLayer(nn.Module):
#    def __init__(self, channel):
#        super(FitSoftLayer, self).__init__()
#
#        body = [
#            nn.Conv2d(channel, channel, 3, 1, 1),
#            nn.GELU(),
#            nn.Conv2d(channel, channel, 3, 1, 1),
#            CALayer(channel),
#        ]
#
#        self.body = nn.Sequential(*body)
#
#    def forward(self, x):
#        f1 = x
#        f1 = self.body(x) + f1
#        return f1
#
#
#class FitSoftNet(nn.Module):
#    def __init__(self, channel=64, num_blocks=5):
#        super(FitSoftNet, self).__init__()
#
#        body = [FitSoftLayer(channel) for _ in range(num_blocks)]
#        self.body = nn.Sequential(*body)
#
#    def forward(self, x):
#        f1 = x
#        f1 = self.body(x) + f1
#        return f1


#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, softmax_scale=10, average=True, res_scale=1, conv=default_conv):
#        super(ENLABlock, self).__init__()
#        self.res_scale = res_scale
#        self.conv_match1 = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.GELU())
#        self.conv_match2 = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.GELU())
#        self.conv_assembly = BasicBlock(conv, channel, channel, 1,bn=False, act=nn.GELU())
#
#        self.dwc = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, groups=channel, padding=7 // 2)
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#        
#        q = self.conv_match1(x)
#        k = self.conv_match2(x)
#        v = self.conv_assembly(x)
#
#        f_q = FS(q)
#        f_k = FS(k)
#
#        f_q = f_q.permute(0,2,3,1).view((N,H*W,C))
#        f_kT = f_k.view(N,C,H*W)
#        v_view = v.view(N,-1,H*W).permute(0,2,1)
#
#        f_kTv = torch.matmul(f_kT, v_view)
#        out = torch.matmul(f_q, f_kTv).permute(0,2,1).view((N,C,H,W))
#
#        v_dwc = self.dwc(v)
#        
#        return out * self.beta + x + v_dwc


class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super(CALayer, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(nf // reduction, nf, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg(x)
        y = self.body(y)
        return torch.mul(x, y)


#class ENLABlock(nn.Module):
#    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
#        super(ENLABlock, self).__init__()
#        
#        self.w1 = nn.Sequential(
#            nn.Linear(channel, channel // 4),
#            nn.GELU(),
#            nn.Linear(channel // 4, channel)
#        )
#
#        self.w2 = nn.Sequential(
#            nn.Linear(channel, channel // 4),
#            nn.GELU(),
#            nn.Linear(channel // 4, channel)
#        )
#
#        self.b = nn.Sequential(
#            nn.Conv2d(channel, channel // 16, 1, 1, 0),
#            nn.GELU(),
#            nn.Conv2d(channel // 16, channel, 1, 1, 0),
#            nn.Sigmoid()
#        )
#
#        self.wv = nn.Conv2d(channel, channel, 1)
#        
#        self.avg = nn.AdaptiveAvgPool2d(1)
#
#        self.dwc = nn.Sequential(
#            nn.Conv2d(channel, channel, 11, stride=1, padding=((11//2)*3), groups=channel, dilation=3)
#        )
#        
#        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
#        
#    def forward(self, x):
#        N,C,H,W = x.shape
#        
#        x_view = x.permute(0,2,3,1).view((N,H*W,C))
#        out = self.w1(x_view)
#
#        x_avg = self.avg(x)
#        b = self.b(x_avg)
#        
#        out = out.permute(0,2,1).view((N,C,H,W)) * b
#        out = self.w2(out.permute(0,2,3,1).view((N,H*W,C)))
#
#        v_dwc = self.dwc(self.wv(x))
#
#        out = out.permute(0,2,1).view((N,C,H,W)) + v_dwc
#        
#        return out * self.beta + x


class ENLABlock(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
        super(ENLABlock, self).__init__()
        
        self.w1 = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, 1),
            nn.GELU(),
            nn.Conv2d(channel // 4, channel, 1, 1)
        )

        self.w2 = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, 1),
            nn.GELU(),
            nn.Conv2d(channel // 4, channel, 1, 1)
        )

        self.w3 = nn.Sequential(
            nn.Conv2d(channel, channel // 4, 1, 1),
            nn.GELU(),
            nn.Conv2d(channel // 4, channel, 1, 1)
        )

        self.b1 = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channel // 16, channel, 1, 1, 0),
            nn.Sigmoid()
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channel // 16, channel, 1, 1, 0),
            nn.Sigmoid()
        )
        
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.wv = nn.Conv2d(channel, channel, 1)


        self.d2dwc_9x9 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=9, groups=channel, padding=((9//2)*2), dilation=2)
        self.dwc_3x3 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, groups=channel, padding=3 // 2)
        self.fusion_18 = nn.Conv2d(2 * channel, channel, 1)

        
        self.d4dwc_9x9 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=9, groups=channel, padding=((9//2)*4), dilation=4)
        self.dwc_5x5 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, groups=channel, padding=5 // 2)
        self.fusion_36 = nn.Conv2d(2 * channel, channel, 1)
        

        self.d6dwc_9x9 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=9, groups=channel, padding=((9//2)*6), dilation=6)
        self.dwc_7x7 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, groups=channel, padding=7 // 2)
        self.fusion_54 = nn.Conv2d(2 * channel, channel, 1)


        self.bypass_1x1 = nn.Conv2d(channel, channel, 1)


        self.v_dwc = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, groups=channel, padding=7 // 2)
        
        self.w_final = nn.Conv2d(4 * channel, channel, 1)
        
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
        
        self.x_ft1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.GELU()
        )

        self.x_ft2 = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.GELU()
        )
        
    def forward(self, x):
        N,C,H,W = x.shape
        
        out = self.w1(x)

        x_avg = self.avg(x)
        
        b1 = self.b1(x_avg)
        b2 = self.b2(x_avg)
        
        out1 = out * b1

        out1 = x + out1
        
        out1 = self.w2(out1)

        v = self.wv(x)
        
        out2 = out1 * b2

        out2 = x + v + out2
        
        out2 = self.w3(out2)

        v_dwc = self.v_dwc(v)

        x_ft1 = self.x_ft1(x) * x

        out = out1 + out2 + v_dwc + x_ft1 + x

        out_d2dwc_9x9 = self.d2dwc_9x9(out)
        out_dwc_3x3 = self.dwc_3x3(out)
        out_18 = [out_d2dwc_9x9, out_dwc_3x3]
        out_18 = torch.cat(out_18, 1)
        out_18 = self.fusion_18(out_18)

        out_d4dwc_9x9 = self.d4dwc_9x9(out)
        out_dwc_5x5 = self.dwc_5x5(out)
        out_36 = [out_d4dwc_9x9, out_dwc_5x5]
        out_36 = torch.cat(out_36, 1)
        out_36 = self.fusion_36(out_36)

        out_d6dwc_9x9 = self.d6dwc_9x9(out)
        out_dwc_7x7 = self.dwc_7x7(out)
        out_54 = [out_d6dwc_9x9, out_dwc_7x7]
        out_54 = torch.cat(out_54, 1)
        out_54 = self.fusion_54(out_54)
        
        out_bypass_1x1 = self.bypass_1x1(out)
        
        dwc_out = [out_18, out_36, out_54, out_bypass_1x1]
        dwc_out = torch.cat(dwc_out, 1)
        dwc_out = self.w_final(dwc_out)
        
        out = dwc_out + out

        x_ft2 = self.x_ft2(x) * x
        
        return out * self.beta + x_ft2 + x
        

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CRB_Layer(nn.Module):
    def __init__(self, nf1):
        super(CRB_Layer, self).__init__()

        self.norm1 = LayerNorm2d(nf1)

        # efficient nonlocal attention
#        self.nla = NonLocalSparseAttention(channels=nf1)
        self.enla = ENLABlock(channel=nf1)

        # local
        conv_local = [
            nn.Conv2d(nf1, nf1, 3, groups=nf1, padding=3 // 2),
            nn.GELU(),
            nn.Conv2d(nf1, nf1, 3, groups=nf1, padding=3 // 2)
        ]

        self.conv_local = nn.Sequential(*conv_local)

        self.conv1_last = nn.Conv2d(2 * nf1, nf1, 1, 1)

        # channel attention
        self.ca = CALayer(nf1)

        self.norm2 = LayerNorm2d(nf1)

        self.gdfn = FeedForward(nf1, 2.66, bias=False)

    def forward(self, x):
        f1 = x

        x = self.norm1(x)

        # pixel attention
        out1 = self.enla(x)

        # local
        out2 = self.conv_local(x)

        out = [out1, out2]
        out = torch.cat(out, 1)

        # the fusion of channel
        out = self.conv1_last(out)

        # channel attention
        out = self.ca(out)

        out_temp = out + f1

        out = self.norm2(out_temp)

        f1 = self.gdfn(out) + out_temp
        return f1


class Restorer(nn.Module):
    def __init__(
        self, in_nc=3, out_nc=3, nf=64, nb=40, scale=4, input_para=10, min=0.0, max=1.0
    ):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        self.head = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)

        # encoder block 1
        encoder_block1 = [CRB_Layer(nf) for _ in range(6)]
        self.encoder_block1 = nn.Sequential(*encoder_block1)

        # downsample 1
        self.down1 = UnetDownsample(nf)

        # encoder block 2
        encoder_block2 = [CRB_Layer(nf * 2) for _ in range(8)]
        self.encoder_block2 = nn.Sequential(*encoder_block2)

        # downsample 2
        self.down2 = UnetDownsample(nf * 2)

        # latent block
        latent_block = [CRB_Layer(nf * 4) for _ in range(10)]
        self.latent_block = nn.Sequential(*latent_block)

        # upsample 1
        self.up1 = UnetUpsample(nf * 4)

        # conv1x1_1
        self.conv1x1_1 = nn.Conv2d(nf * 2 * 2, nf * 2, 1, 1)

        # decoder block 1
        decoder_block1 = [CRB_Layer(nf * 2) for _ in range(6)]
        self.decoder_block1 = nn.Sequential(*decoder_block1)

        # upsample 2
        self.up2 = UnetUpsample(nf * 2)

        # decoder block 2
        decoder_block2 = [CRB_Layer(nf * 2) for _ in range(8)]
        self.decoder_block2 = nn.Sequential(*decoder_block2)

        # refine block
        refine_block = [CRB_Layer(nf * 2) for _ in range(8)]
        self.refine_block = nn.Sequential(*refine_block)

        self.fusion = nn.Conv2d(nf * 2, nf, 3, 1, 1)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, 3, 3, 1, 1),
            )
        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale ** 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, 3, 3, 1, 1),
            )

    def forward(self, input):
        B, C, H, W = input.size()  # I_LR batch

        # bicubic
        upsample = transforms.Resize((input.shape[2] * 4, input.shape[3] * 4), interpolation=transforms.InterpolationMode.BICUBIC)
        lr_bic = upsample(input)
        lr_bic = torch.clamp(lr_bic, min=0, max=1)

        f = self.head(input)

        # encoder
        f_en1 = self.encoder_block1(f)
        f_en1_down = self.down1(f_en1)

        f_en2 = self.encoder_block2(f_en1_down)
        f_en2_down = self.down2(f_en2)

        # latent
        f_latent = self.latent_block(f_en2_down)

        # decoder
        f_latent_up = self.up1(f_latent)
        f_conv1 = self.conv1x1_1(torch.cat([f_en2, f_latent_up], dim=1))
        f_de1 = self.decoder_block1(f_conv1)

        f_de1_up = self.up2(f_de1)
        f_cat2 = torch.cat([f_en1, f_de1_up], dim=1)
        f_de2 = self.decoder_block2(f_cat2)

        # refine
        f_re = self.refine_block(f_de2)

        f = self.fusion(f_re)
        out = self.upscale(f) + lr_bic

        return out  # torch.clamp(out, min=self.min, max=self.max)
        


from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class LabNet_STEA(nn.Module):
    def __init__(
        self,
        nf=64,
        nb=40,
        upscale=4,
        input_para=10,
        kernel_size=21,
        loop=1
    ):
        super(LabNet_STEA, self).__init__()

        self.ksize = kernel_size
        self.scale = upscale

        self.Restorer = Restorer(nf=nf, nb=nb, scale=self.scale, input_para=input_para)

    def forward(self, lr):

        B, C, H, W = lr.shape
        sr = self.Restorer(lr)
        return sr
        