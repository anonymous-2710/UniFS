import torch
from torch import nn
import math
import torch.nn.functional as F
from einops import rearrange


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class invPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        tensor = tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4)
        return tensor.contiguous().view(b, -1, y // ratio, x // ratio)


class UpSampler(nn.Sequential):
    def __init__(self, scale, n_feats):

        m = []
        if scale == 8:
            kernel_size = 3
        elif scale == 16:
            kernel_size = 5
        else:
            kernel_size = 1

        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * n_feats, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2))
                m.append(nn.PixelShuffle(upscale_factor=2))
                m.append(nn.PReLU())
        super(UpSampler, self).__init__(*m)


class InvUpSampler(nn.Sequential):
    def __init__(self, scale, n_feats):

        m = []
        if scale == 8:
            kernel_size = 3
        elif scale == 16:
            kernel_size = 5
        else:
            kernel_size = 1
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(invPixelShuffle(2))
                m.append(nn.Conv2d(in_channels=n_feats * 4, out_channels=n_feats, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2))
                m.append(nn.PReLU())
        super(InvUpSampler, self).__init__(*m)


class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)


class ConvBNReLU2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, act=None, norm=None):
        super(ConvBNReLU2D, self).__init__()

        self.layers = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.act = None
        self.norm = None
        if norm == 'BN':
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm = torch.nn.InstanceNorm2d(out_channels)
        elif norm == 'GN':
            self.norm = torch.nn.GroupNorm(2, out_channels)
        elif norm == 'WN':
            self.layers = torch.nn.utils.weight_norm(self.layers)
        elif norm == 'Adaptive':
            self.norm = AdaptiveNorm(n=out_channels)

        if act == 'PReLU':
            self.act = torch.nn.PReLU()
        elif act == 'SELU':
            self.act = torch.nn.SELU(True)
        elif act == 'LeakyReLU':
            self.act = torch.nn.LeakyReLU(negative_slope=0.02, inplace=True)
        elif act == 'ELU':
            self.act = torch.nn.ELU(inplace=True)
        elif act == 'ReLU':
            self.act = torch.nn.ReLU(True)
        elif act == 'Tanh':
            self.act = torch.nn.Tanh()
        elif act == 'Sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'SoftMax':
            self.act = torch.nn.Softmax2d()

    def forward(self, inputs):

        out = self.layers(inputs)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, act='ReLU', padding=1),
            ConvBNReLU2D(num_features, out_channels=num_features, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        return F.relu(self.layers(inputs) + inputs)


class DownSample(nn.Module):
    def __init__(self, num_features, act, norm, scale=2):
        super(DownSample, self).__init__()
        if scale == 1:
            self.layers = nn.Sequential(
                ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, act=act, norm=norm,
                             padding=1),
                ConvBNReLU2D(in_channels=num_features * scale * scale, out_channels=num_features, kernel_size=1,
                             act=act, norm=norm)
            )
        else:
            self.layers = nn.Sequential(
                ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, act=act, norm=norm,
                             padding=1),
                invPixelShuffle(ratio=scale),
                ConvBNReLU2D(in_channels=num_features * scale * scale, out_channels=num_features, kernel_size=1,
                             act=act, norm=norm)
            )

    def forward(self, inputs):
        return self.layers(inputs)


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, norm, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            ResBlock(n_feat) for _ in range(n_resblocks)]

        modules_body.append(ConvBNReLU2D(n_feat, n_feat, kernel_size, padding=1, act=act, norm=norm))
        self.body = nn.Sequential(*modules_body)
        self.re_scale = Scale(1)

    def forward(self, x):
        res = self.body(x)
        return res + self.re_scale(x)


class FreBlock9(nn.Module):
    def __init__(self, channels):
        super(FreBlock9, self).__init__()

        self.fpre = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                      nn.Conv2d(channels, channels, 3, 1, 1))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        # print("x: ", x.shape)
        _, _, H, W = x.shape
        msF = torch.fft.rfft2(self.fpre(x) + 1e-8, norm='backward')

        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        # print("msf_amp: ", msF_amp.shape)
        amp_fuse = self.amp_fuse(msF_amp)
        # print(amp_fuse.shape, msF_amp.shape)
        amp_fuse = amp_fuse + msF_amp
        pha_fuse = self.pha_fuse(msF_pha)
        pha_fuse = pha_fuse + msF_pha

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))
        out = self.post(out)
        out = out + x
        out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
        # print("out: ", out.shape)
        return out


class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FuseBlock7(nn.Module):
    def __init__(self, channels):
        super(FuseBlock7, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels)
        self.spa_att = Attention(dim=channels)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1),
                                  nn.Conv2d(channels, 2 * channels, 3, 1, 1), nn.Sigmoid())

    def forward(self, spa, fre):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        fre = self.fre_att(fre, spa) + fre
        spa = self.fre_att(spa, fre) + spa
        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


class FuseBlock6(nn.Module):
    def __init__(self, channels):
        super(FuseBlock6, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1),
                                  nn.Conv2d(channels, 2 * channels, 3, 1, 1), nn.Sigmoid())

    def forward(self, spa, fre):
        fre = self.fre(fre)
        spa = self.spa(spa)

        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


class FuseBlock4(nn.Module):
    def __init__(self, channels):
        super(FuseBlock4, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1), nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, spa, fre):
        fre = self.fre(fre)
        spa = self.spa(spa)

        fuse = self.fuse(torch.cat((fre, spa), 1))
        res = torch.nan_to_num(fuse, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


class Modality_FuseBlock4(nn.Module):
    def __init__(self, channels):
        super(FuseBlock4, self).__init__()
        self.t1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.t2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1), nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, t1, t2):
        t1 = self.t1(t1)
        t2 = self.t2(t2)

        fuse = self.fuse(torch.cat((t1, t2), 1))
        res = torch.nan_to_num(fuse, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


class Modality_FuseBlock7(nn.Module):
    def __init__(self, channels):
        super(Modality_FuseBlock7, self).__init__()
        self.t1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.t2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.t1_att = Attention(dim=channels)
        self.t2_att = Attention(dim=channels)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1),
                                  nn.Conv2d(channels, 2 * channels, 3, 1, 1), nn.Sigmoid())

    def forward(self, t1, t2):
        t1 = self.t1(t1)
        t2 = self.t2(t2)
        t1 = self.t1_att(t1, t2) + t1
        t2 = self.t2_att(t2, t1) + t2
        fuse = self.fuse(torch.cat((t1, t2), 1))
        t1_a, t2_a = fuse.chunk(2, dim=1)
        t2 = t2_a * t2
        t1 = t1 * t1_a
        res = t1 + t2

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


class Modality_FuseBlock6(nn.Module):
    def __init__(self, channels):
        super(Modality_FuseBlock6, self).__init__()
        self.t1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.t2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1),
                                  nn.Conv2d(channels, 2 * channels, 3, 1, 1), nn.Sigmoid())

    def forward(self, t1, t2):
        t1 = self.t1(t1)
        t2 = self.t2(t2)

        fuse = self.fuse(torch.cat((t1, t2), 1))
        t1_a, t2_a = fuse.chunk(2, dim=1)
        t2 = t2_a * t2
        t1 = t1 * t1_a
        res = t1 + t2

        res = torch.nan_to_num(res, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


class Modality_FuseBlock4(nn.Module):
    def __init__(self, channels):
        super(Modality_FuseBlock4, self).__init__()
        self.t1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.t2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1), nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, t1, t2):
        t1 = self.t1(t1)
        t2 = self.t2(t2)

        fuse = self.fuse(torch.cat((t1, t2), 1))
        res = torch.nan_to_num(fuse, nan=1e-5, posinf=1e-5, neginf=1e-5)
        return res


class TwoBranch(nn.Module):
    def __init__(self, base_num_every_group, num_features, num_channels, act):
        super(TwoBranch, self).__init__()

        num_group = 4
        num_every_group = base_num_every_group

        self.init_T2_frq_branch(num_features, num_channels, act)
        self.init_T2_spa_branch(num_features, num_channels, act, num_every_group)
        self.init_T2_fre_spa_fusion(num_features)

        self.init_T1_frq_branch(num_features, act)
        self.init_T1_spa_branch(num_features, act, num_every_group)

        self.init_modality_fre_fusion(num_features)
        self.init_modality_spa_fusion(num_features)

    def init_T2_frq_branch(self, num_features, num_channels, act):
        ### T2frequency branch
        modules_head_fre = [ConvBNReLU2D(1, out_channels=num_features,
                                         kernel_size=3, padding=1, act=act)]
        self.head_fre = nn.Sequential(*modules_head_fre)

        modules_down1_fre = [DownSample(num_features, False, False),
                             FreBlock9(num_features)
                             ]

        self.down1_fre = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo = nn.Sequential(FreBlock9(num_features))

        modules_down2_fre = [DownSample(num_features, False, False),
                             FreBlock9(num_features)
                             ]
        self.down2_fre = nn.Sequential(*modules_down2_fre)

        self.down2_fre_mo = nn.Sequential(FreBlock9(num_features))

        modules_down3_fre = [DownSample(num_features, False, False),
                             FreBlock9(num_features)
                             ]
        self.down3_fre = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo = nn.Sequential(FreBlock9(num_features))

        modules_neck_fre = [FreBlock9(num_features)
                            ]
        self.neck_fre = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo = nn.Sequential(FreBlock9(num_features))

        modules_up1_fre = [UpSampler(2, num_features),
                           FreBlock9(num_features)
                           ]
        self.up1_fre = nn.Sequential(*modules_up1_fre)
        self.up1_fre_mo = nn.Sequential(FreBlock9(num_features))

        modules_up2_fre = [UpSampler(2, num_features),
                           FreBlock9(num_features)
                           ]
        self.up2_fre = nn.Sequential(*modules_up2_fre)
        self.up2_fre_mo = nn.Sequential(FreBlock9(num_features))

        modules_up3_fre = [UpSampler(2, num_features),
                           FreBlock9(num_features)
                           ]
        self.up3_fre = nn.Sequential(*modules_up3_fre)
        self.up3_fre_mo = nn.Sequential(FreBlock9(num_features))

        # define tail module
        modules_tail_fre = [
            ConvBNReLU2D(num_features, out_channels=num_channels, kernel_size=3, padding=1,
                         act=act)]
        self.tail_fre = nn.Sequential(*modules_tail_fre)

    def init_T2_spa_branch(self, num_features, num_channels, act, num_every_group):
        ### spatial branch
        modules_head = [ConvBNReLU2D(1, out_channels=num_features,
                                     kernel_size=3, padding=1, act=act)]
        self.head = nn.Sequential(*modules_head)

        modules_down1 = [DownSample(num_features, False, False),
                         ResidualGroup(
                             num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down1 = nn.Sequential(*modules_down1)

        self.down1_mo = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_down2 = [DownSample(num_features, False, False),
                         ResidualGroup(
                             num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down2 = nn.Sequential(*modules_down2)

        self.down2_mo = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_down3 = [DownSample(num_features, False, False),
                         ResidualGroup(
                             num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down3 = nn.Sequential(*modules_down3)
        self.down3_mo = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_neck = [ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
        ]
        self.neck = nn.Sequential(*modules_neck)

        self.neck_mo = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_up1 = [UpSampler(2, num_features),
                       ResidualGroup(
                           num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
                       ]
        self.up1 = nn.Sequential(*modules_up1)

        self.up1_mo = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_up2 = [UpSampler(2, num_features),
                       ResidualGroup(
                           num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
                       ]
        self.up2 = nn.Sequential(*modules_up2)
        self.up2_mo = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_up3 = [UpSampler(2, num_features),
                       ResidualGroup(
                           num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
                       ]
        self.up3 = nn.Sequential(*modules_up3)
        self.up3_mo = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        # define tail module
        modules_tail = [
            ConvBNReLU2D(num_features, out_channels=num_channels, kernel_size=3, padding=1,
                         act=act)]

        self.tail = nn.Sequential(*modules_tail)

    def init_T2_fre_spa_fusion(self, num_features):
        ### T2 frq & spa fusion part
        conv_fuse = []
        for i in range(14):
            conv_fuse.append(FuseBlock7(num_features))
        self.conv_fuse = nn.Sequential(*conv_fuse)

    def init_T1_frq_branch(self, num_features, act):
        ### T2frequency branch
        modules_head_fre = [ConvBNReLU2D(1, out_channels=num_features,
                                         kernel_size=3, padding=1, act=act)]
        self.head_fre_T1 = nn.Sequential(*modules_head_fre)

        modules_down1_fre = [DownSample(num_features, False, False),
                             FreBlock9(num_features)
                             ]

        self.down1_fre_T1 = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo_T1 = nn.Sequential(FreBlock9(num_features))

        modules_down2_fre = [DownSample(num_features, False, False),
                             FreBlock9(num_features)
                             ]
        self.down2_fre_T1 = nn.Sequential(*modules_down2_fre)

        self.down2_fre_mo_T1 = nn.Sequential(FreBlock9(num_features))

        modules_down3_fre = [DownSample(num_features, False, False),
                             FreBlock9(num_features)
                             ]
        self.down3_fre_T1 = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo_T1 = nn.Sequential(FreBlock9(num_features))

        modules_neck_fre = [FreBlock9(num_features)
                            ]
        self.neck_fre_T1 = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo_T1 = nn.Sequential(FreBlock9(num_features))

    def init_T1_spa_branch(self, num_features, act, num_every_group):
        ### spatial branch
        modules_head = [ConvBNReLU2D(1, out_channels=num_features,
                                     kernel_size=3, padding=1, act=act)]
        self.head_T1 = nn.Sequential(*modules_head)

        modules_down1 = [DownSample(num_features, False, False),
                         ResidualGroup(
                             num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down1_T1 = nn.Sequential(*modules_down1)

        self.down1_mo_T1 = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_down2 = [DownSample(num_features, False, False),
                         ResidualGroup(
                             num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down2_T1 = nn.Sequential(*modules_down2)

        self.down2_mo_T1 = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_down3 = [DownSample(num_features, False, False),
                         ResidualGroup(
                             num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down3_T1 = nn.Sequential(*modules_down3)
        self.down3_mo_T1 = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

        modules_neck = [ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None)
        ]
        self.neck_T1 = nn.Sequential(*modules_neck)

        self.neck_mo_T1 = nn.Sequential(ResidualGroup(
            num_features, 3, 4, act=act, n_resblocks=num_every_group, norm=None))

    def init_modality_fre_fusion(self, num_features):
        conv_fuse = []
        for i in range(5):
            conv_fuse.append(Modality_FuseBlock6(num_features))
        self.conv_fuse_fre = nn.Sequential(*conv_fuse)

    def init_modality_spa_fusion(self, num_features):
        conv_fuse = []
        for i in range(5):
            conv_fuse.append(Modality_FuseBlock6(num_features))
        self.conv_fuse_spa = nn.Sequential(*conv_fuse)

    def forward(self, main, aux, **kwargs):
        #### T1 fre encoder
        t1_fre = self.head_fre_T1(aux)  # 128

        down1_fre_t1 = self.down1_fre_T1(t1_fre)  # 64
        down1_fre_mo_t1 = self.down1_fre_mo_T1(down1_fre_t1)

        down2_fre_t1 = self.down2_fre_T1(down1_fre_mo_t1)  # 32
        down2_fre_mo_t1 = self.down2_fre_mo_T1(down2_fre_t1)

        down3_fre_t1 = self.down3_fre_T1(down2_fre_mo_t1)  # 16
        down3_fre_mo_t1 = self.down3_fre_mo_T1(down3_fre_t1)

        neck_fre_t1 = self.neck_fre_T1(down3_fre_mo_t1)  # 16
        neck_fre_mo_t1 = self.neck_fre_mo_T1(neck_fre_t1)

        #### T2 fre encoder and T1 & T2 fre fusion
        x_fre = self.head_fre(main)  # 128
        x_fre_fuse = self.conv_fuse_fre[0](t1_fre, x_fre)

        down1_fre = self.down1_fre(x_fre_fuse)  # 64
        down1_fre_mo = self.down1_fre_mo(down1_fre)
        down1_fre_mo_fuse = self.conv_fuse_fre[1](down1_fre_mo_t1, down1_fre_mo)

        down2_fre = self.down2_fre(down1_fre_mo_fuse)  # 32
        down2_fre_mo = self.down2_fre_mo(down2_fre)
        down2_fre_mo_fuse = self.conv_fuse_fre[2](down2_fre_mo_t1, down2_fre_mo)

        down3_fre = self.down3_fre(down2_fre_mo_fuse)  # 16
        down3_fre_mo = self.down3_fre_mo(down3_fre)
        down3_fre_mo_fuse = self.conv_fuse_fre[3](down3_fre_mo_t1, down3_fre_mo)

        neck_fre = self.neck_fre(down3_fre_mo_fuse)  # 16
        neck_fre_mo = self.neck_fre_mo(neck_fre)
        neck_fre_mo_fuse = self.conv_fuse_fre[4](neck_fre_mo_t1, neck_fre_mo)

        #### T2 fre decoder
        neck_fre_mo = neck_fre_mo_fuse + down3_fre_mo_fuse

        up1_fre = self.up1_fre(neck_fre_mo)  # 32
        up1_fre_mo = self.up1_fre_mo(up1_fre)
        up1_fre_mo = up1_fre_mo + down2_fre_mo_fuse

        up2_fre = self.up2_fre(up1_fre_mo)  # 64
        up2_fre_mo = self.up2_fre_mo(up2_fre)
        up2_fre_mo = up2_fre_mo + down1_fre_mo_fuse

        up3_fre = self.up3_fre(up2_fre_mo)  # 128
        up3_fre_mo = self.up3_fre_mo(up3_fre)
        up3_fre_mo = up3_fre_mo + x_fre_fuse

        res_fre = self.tail_fre(up3_fre_mo)

        #### T1 spa encoder
        x_t1 = self.head_T1(aux)  # 128

        down1_t1 = self.down1_T1(x_t1)  # 64
        down1_mo_t1 = self.down1_mo_T1(down1_t1)

        down2_t1 = self.down2_T1(down1_mo_t1)  # 32
        down2_mo_t1 = self.down2_mo_T1(down2_t1)  # 32

        down3_t1 = self.down3_T1(down2_mo_t1)  # 16
        down3_mo_t1 = self.down3_mo_T1(down3_t1)  # 16

        neck_t1 = self.neck_T1(down3_mo_t1)  # 16
        neck_mo_t1 = self.neck_mo_T1(neck_t1)

        #### T2 spa encoder and fusion
        x = self.head(main)  # 128

        x_fuse = self.conv_fuse_spa[0](x_t1, x)
        down1 = self.down1(x_fuse)  # 64
        down1_fuse = self.conv_fuse[0](down1_fre, down1)
        down1_mo = self.down1_mo(down1_fuse)
        down1_fuse_mo = self.conv_fuse[1](down1_fre_mo_fuse, down1_mo)

        down1_fuse_mo_fuse = self.conv_fuse_spa[1](down1_mo_t1, down1_fuse_mo)
        down2 = self.down2(down1_fuse_mo_fuse)  # 32
        down2_fuse = self.conv_fuse[2](down2_fre, down2)
        down2_mo = self.down2_mo(down2_fuse)  # 32
        down2_fuse_mo = self.conv_fuse[3](down2_fre_mo, down2_mo)

        down2_fuse_mo_fuse = self.conv_fuse_spa[2](down2_mo_t1, down2_fuse_mo)
        down3 = self.down3(down2_fuse_mo_fuse)  # 16
        down3_fuse = self.conv_fuse[4](down3_fre, down3)
        down3_mo = self.down3_mo(down3_fuse)  # 16
        down3_fuse_mo = self.conv_fuse[5](down3_fre_mo, down3_mo)

        down3_fuse_mo_fuse = self.conv_fuse_spa[3](down3_mo_t1, down3_fuse_mo)
        neck = self.neck(down3_fuse_mo_fuse)  # 16
        neck_fuse = self.conv_fuse[6](neck_fre, neck)
        neck_mo = self.neck_mo(neck_fuse)
        neck_mo = neck_mo + down3_mo
        neck_fuse_mo = self.conv_fuse[7](neck_fre_mo, neck_mo)

        neck_fuse_mo_fuse = self.conv_fuse_spa[4](neck_mo_t1, neck_fuse_mo)
        #### T2 spa decoder
        up1 = self.up1(neck_fuse_mo_fuse)  # 32
        up1_fuse = self.conv_fuse[8](up1_fre, up1)
        up1_mo = self.up1_mo(up1_fuse)
        up1_mo = up1_mo + down2_mo
        up1_fuse_mo = self.conv_fuse[9](up1_fre_mo, up1_mo)

        up2 = self.up2(up1_fuse_mo)  # 64
        up2_fuse = self.conv_fuse[10](up2_fre, up2)
        up2_mo = self.up2_mo(up2_fuse)
        up2_mo = up2_mo + down1_mo
        up2_fuse_mo = self.conv_fuse[11](up2_fre_mo, up2_mo)

        up3 = self.up3(up2_fuse_mo)  # 128

        up3_fuse = self.conv_fuse[12](up3_fre, up3)
        up3_mo = self.up3_mo(up3_fuse)

        up3_mo = up3_mo + x
        up3_fuse_mo = self.conv_fuse[13](up3_fre_mo, up3_mo)
        # import matplotlib.pyplot as plt
        # plt.axis('off')
        # plt.imshow((255*up3_fre_mo[0].detach().cpu().numpy()[0]))
        # plt.savefig('up3_fre_mo.jpg', bbox_inches='tight', pad_inches=0)
        # plt.clf() 

        # plt.axis('off')
        # plt.imshow((255*up3_mo[0].detach().cpu().numpy()[0]))
        # plt.savefig('up3_mo.jpg', bbox_inches='tight', pad_inches=0)
        # plt.clf() 

        # plt.axis('off')
        # plt.imshow((255*up3_fuse_mo[0].detach().cpu().numpy()[0]))
        # plt.savefig('up3_fuse_mo.jpg', bbox_inches='tight', pad_inches=0)
        # plt.clf() 
        # breakpoint()

        res = self.tail(up3_fuse_mo)

        return {'img_out': res + main, 'img_fre': res_fre + main}


if __name__ == '__main__':
    pd_hr = torch.randn(1, 1, 200, 200)
    t2_lr = torch.randn(1, 1, 200, 200)

    model = TwoBranch(base_num_every_group=1, num_features=64, num_channels=1, act='PReLU')
    out = model(t2_lr, pd_hr)

    img_out = out['img_out']
    print(img_out.shape)
    target = torch.randn_like(img_out)
    loss = nn.MSELoss()(img_out, target)
    loss.backward()
    print(f"Loss gradient: {loss.grad_fn}")
