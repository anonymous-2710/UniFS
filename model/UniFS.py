import torch
from torch import nn
from RCAN import default_conv, ResidualGroup, Upsampler

class RCANEncodeyPart(nn.Module):
    def __init__(self,
                 input_nc=1,
                 n_resgroups=10,  # RCAN n_resgroups
                 n_resblocks=20,  # RCAN n_resblocks
                 n_feats=64,  # RCAN n_feats
                 reduction=16,  # number of feature maps reduction
                 rgb_range=255,  # maximum value of RGB
                 res_scale=1,  # residual scaling
                 kernel_size=3,
                 act=nn.ReLU(True),
                 conv=default_conv):
        super(RCANEncodeyPart, self).__init__()

        # RGB mean for DIV2K
        # self.sub_mean = MeanShift_gray(rgb_range)

        # define head module
        modules_head = [conv(input_nc, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        return x, self.body
    
class TwoInputRCAN(nn.Module):
    def __init__(self,
                 input_nc=1,
                 output_nc=1,
                 n_resgroups=10,  # RCAN n_resgroups
                 n_resblocks=20,  # RCAN n_resblocks
                 n_feats=64,  # RCAN n_feats
                 chop=True,  # RCAN chop
                 reduction=16,  # number of feature maps reduction
                 rgb_range=1,  # maximum value of RGB
                 res_scale=1,  # residual scaling
                 conv=default_conv,
                 concat_index_list=[3, 6, 10],
                 bias=True,
                 **kwargs):
        super(TwoInputRCAN, self).__init__()
        self.final_output = None
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.reduction = reduction
        self.chop = chop
        self.rgb_range = rgb_range
        self.res_scale = res_scale
        self.conv = default_conv
        self.concat_index_list = concat_index_list
        self.bias = bias
        self.kernel_size = 3
        self.scale = 1
        self.act = nn.ReLU(True)
        self.modules_tail = [Upsampler(conv, self.scale, n_feats, act=False),
                             conv(n_feats, output_nc, self.kernel_size)]
        # self.add_mean = MeanShift_gray(rgb_range, sign=1)
        self.t2_encoder = RCANEncodeyPart(input_nc, n_resgroups, n_resblocks, n_feats,
                                          reduction, rgb_range, res_scale, self.kernel_size, self.act, conv)
        self.pd_encoder = RCANEncodeyPart(input_nc, n_resgroups, n_resblocks, n_feats,
                                          reduction, rgb_range, res_scale, self.kernel_size, self.act, conv)
        self.tail = nn.Sequential(*self.modules_tail)
        self.concat_index_list = concat_index_list
        # self.reduce_chan_concat = nn.ModuleList([nn.Conv2d(int(n_feats * 2), int(n_feats), kernel_size=1, bias=bias)
        #                                          for _ in concat_index_list])  # changed after RCAN_Multix4_0116
        self.reduce_chan_concat = nn.ModuleList(
            [nn.Conv2d(int(n_feats * 2), int(n_feats), kernel_size=3, bias=bias, padding=1) for _ in concat_index_list])
        ignore_grad(self.pd_encoder.body, max(concat_index_list))

    def forward(self, pd_hr, t2_lr, **kwargs):
        # sum shift and head part
        pd_before_head, pd_body = self.pd_encoder(pd_hr)
        t2_before_head, t2_body = self.t2_encoder(t2_lr)

        # body part
        pd_hr_x, t2_lr_x = pd_before_head, t2_before_head
        reduce_chan_idx = 0
        for idx, (pd_layer, t2_layer) in enumerate(zip(pd_body.children(), t2_body.children())):
            pd_hr_x = pd_layer(pd_hr_x)
            t2_lr_x = t2_layer(t2_lr_x)
            if idx in self.concat_index_list:
                pd_t2_concat = torch.cat([pd_hr_x, t2_lr_x], 1)
                t2_lr_x = self.reduce_chan_concat[reduce_chan_idx](pd_t2_concat)
                reduce_chan_idx += 1

        t2_lr_x = t2_lr_x + t2_before_head

        # tail part
        self.final_output = self.tail(t2_lr_x)
        # t2_lr_x = self.add_mean(t2_lr_x)
        return self.final_output

def ignore_grad(body, idx):
    for i, module in enumerate(body):
        if i > idx:
            for param in module.parameters():
                param.requires_grad = False

def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class Fre_Spatial_fuse(nn.Module):
    def __init__(self, channels):
        super(Fre_Spatial_fuse, self).__init__()
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.cha_att = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels * 2, channels, 3, 1, 1)

    def forward(self, frefuse, spafuse):
        spa_map = self.spa_att(spafuse - frefuse)
        spa_res = frefuse * spa_map + spafuse
        cat_f = torch.cat([spa_res, frefuse], 1)
        cha_res = self.post(self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f)

        return cha_res
    
class Spatial_Fuse(nn.Module):
    def __init__(self, channels):
        super(Spatial_Fuse, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.spatial_fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                          nn.Conv2d(channels, channels, 1, 1, 0))

    def forward(self, x, x_ref):
        spatial_fuse = self.spatial_fuse(torch.cat([x, x_ref], dim=1))
        return spatial_fuse

class SFIB_With_MaskPrompt(nn.Module):
    def __init__(self, channels):
        super(SFIB_With_MaskPrompt, self).__init__()
        self.spatial_fuse = Spatial_Fuse(channels)
        self.frequency_fuse = Frequency_Fuse_With_MaskPrompt(channels)
        self.fre_spa_fuse = Fre_Spatial_fuse(channels)

    def forward(self, t2_lr_x, pd_hr_x, mask):
        frefuse = self.frequency_fuse(t2_lr_x, pd_hr_x, mask)
        spafuse = self.spatial_fuse(t2_lr_x, pd_hr_x)
        cha_res = self.fre_spa_fuse(frefuse, spafuse)
        return cha_res

class SpatialPromptGenerator(nn.Module):
    def __init__(self, feat_dim=64):
        super().__init__()
        # 动态权重生成器
        self.fuse = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        # Mask特征增强
        self.mask_proj = nn.Sequential(
            nn.Conv2d(1, feat_dim // 2, 3, padding=1),  # 降维
            nn.GELU(),
            nn.Conv2d(feat_dim // 2, feat_dim, 3, padding=1)  # 升维对齐
        )
        self.adjust_conv = nn.Conv2d(feat_dim, feat_dim, 1)

    def forward(self, feat, mask):
        """
        feat: 高频特征 (B,64,H,W)
        mask: k-space mask (B,1,H,W)
        """
        # 通道维度统计量（修正点）
        avg_out = torch.mean(feat, dim=1, keepdim=True)  # (B,1,H,W)
        max_out, _ = torch.max(feat, dim=1, keepdim=True)  # (B,1,H,W)
        # 动态融合
        stats = torch.cat([avg_out, max_out], dim=1)  # (B,2,H,W)
        spatial_weight = self.fuse(stats)  # (B,1,H,W)
        # Mask调制（物理约束增强）
        mask_feat = self.mask_proj(mask)  # (B,64,H,W)
        prompt = mask_feat * spatial_weight  # 空间自适应调制
        return self.adjust_conv(prompt)

class Frequency_Fuse_With_MaskPrompt(nn.Module):
    def __init__(self, channels):
        super(Frequency_Fuse_With_MaskPrompt, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(3 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(3 * channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

        self.amp_prompt = SpatialPromptGenerator(channels)
        self.pha_prompt = SpatialPromptGenerator(channels)

    def forward(self, msf, panf, mask):
        _, _, H, W = msf.shape
        msF = torch.fft.fft2(self.pre1(msf) + 1e-8, norm='backward')
        panF = torch.fft.fft2(self.pre2(panf) + 1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)

        prompt_amp = self.amp_prompt(msF_amp, mask)
        prompt_pha = self.pha_prompt(msF_pha, mask)

        amp_fuse = self.amp_fuse(torch.cat([msF_amp, panF_amp, prompt_amp], 1))
        pha_fuse = self.pha_fuse(torch.cat([msF_pha, panF_pha, prompt_pha], 1))

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.ifft2(out))

        return self.post(out)

class UniFS(TwoInputRCAN):
    def __init__(self, **kwargs):
        super(UniFS, self).__init__(**kwargs)
        channels = self.n_feats
        self.multi_feature_fuse = nn.ModuleList([SFIB_With_MaskPrompt(channels) for _ in self.concat_index_list])

    def forward(self, pd_hr, t2_lr, **kwargs):
        # sum shift and head part
        mask = kwargs['mask']
        pd_before_head, pd_body = self.pd_encoder(pd_hr)
        t2_before_head, t2_body = self.t2_encoder(t2_lr)

        pd_hr_x, t2_lr_x = pd_before_head, t2_before_head
        fuse_idx = 0
        for idx, (pd_layer, t2_layer) in enumerate(
                zip(pd_body.children(), t2_body.children())):
            pd_hr_x = pd_layer(pd_hr_x)
            t2_lr_x = t2_layer(t2_lr_x)
            if idx in self.concat_index_list:
                cha_res = self.multi_feature_fuse[fuse_idx](t2_lr_x, pd_hr_x, mask)
                t2_lr_x = t2_lr_x + cha_res
                fuse_idx += 1

        assert fuse_idx == len(self.concat_index_list)
        t2_lr_x = t2_lr_x + t2_before_head

        # tail part
        t2_lr_x = self.tail(t2_lr_x)
        # t2_lr_x = self.add_mean(t2_lr_x)
        return t2_lr_x

if __name__ == '__main__':
    pd_hr = torch.randn(1, 1, 200, 200)
    t2_lr = torch.randn(1, 1, 200, 200)
    mask = torch.randn(1, 1, 200, 200)
    model = UniFS(n_resgroups=3, n_resblocks=6, concat_index_list=[0, 1, 2])
    out = model(pd_hr, t2_lr, mask=mask)
    print(out.shape)
    target = torch.randn_like(out)
    loss = nn.MSELoss()(out, target)
    loss.backward()
    print(f"Loss gradient: {loss.grad_fn}")
