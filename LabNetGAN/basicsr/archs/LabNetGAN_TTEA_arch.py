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


class TTEA(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=1, scale=3, stride=1, average=True, conv=default_conv, height=None, weight=None):
        super(TTEA, self).__init__()
        
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

        self.b3 = nn.Sequential(
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
        b3 = self.b3(x_avg)

        v = self.wv(x)
        
        out1_temp = out * b1

        out1 = x + out1_temp + v

        # FTEA
        out1 = self.w3(out1)
        
        out2 = self.w2(out1_temp)
        
        out2_temp = out2 * b2

        out2 = x + out2_temp + v 

        # STEA
        out2 = self.w3(out2)

        out3 = self.w2(out2_temp)
        
        out3_temp = out3 * b3

        out3 = x + out3_temp + v

        # TTEA
        out3 = self.w3(out3)

        # DWC(XWv)
        v_dwc = self.v_dwc(v)

        x_ft1 = self.x_ft1(x) * x

        # FTEA + STEA + TTEA + DWC(XWv) + ...
        out = out1 + out2 + out3 + v_dwc + x_ft1 + x

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

        # TTEA
        self.TTEA = TTEA(channel=nf1)

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

        # TTEA
        out1 = self.TTEA(x)

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
class LabNet_TTEA(nn.Module):
    def __init__(
        self,
        nf=64,
        nb=40,
        upscale=4,
        input_para=10,
        kernel_size=21,
        loop=0,
        pca_matrix_path=None,
    ):
        super(LabNet_TTEA, self).__init__()

        self.ksize = kernel_size
        self.scale = upscale

        self.Restorer = Restorer(nf=nf, nb=nb, scale=self.scale, input_para=input_para)

    def forward(self, lr):

        B, C, H, W = lr.shape
        sr = self.Restorer(lr)
        return sr
        