"""SCAN: Spatial Color Attention Network.

This module is implementation of 'SCAN: Spatial Color Attention Networks
for Real Single Image Super-Resolution' published by Xu and Li in CVPR
Workshop in 2019 June.

SCAN allows any input and output dimension. Only number of base network
layers is restricted to 3.

Todo:
    - Hack: BaseNet should be written using for statement.
"""
import torch
import torch.nn as nn

from . import rcan

class SpatialAttention(nn.Module):
    """Spatial Attention.
    
    This class implements spatial attention layer consists of
    conv layer, activation function, conv layer.

    Note:
        According to the thesis, number of conv layer's kernel is 1.
        This means feature map is changed like:
            C ->[conv]-> 1 ->[relu]-> 1 ->[conv]-> 1
        However, 2nd conv layer will not do anything in this case.
        So it would be better to use larger number of 1st layer's
        output channel (specified by channel_mid) than 1.
    """
    def __init__(self, channel, channel_mid, act=nn.ReLU(inplace=True)):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channel, channel_mid, kernel_size=1, padding=0, bias=True),
            act,
            nn.Conv2d(channel_mid, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
            )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.tensor(B,C,H,W)): Feature map.

        Returns:
            torch.tensor(B,1,H,W): Spatial attention map.
        """
        x = self.body(x)
        return x


class RCSA(nn.Module):
    """Residual Channel-Spatial Attention (RCSA).

    This class implements main network including channel and spatial
    attention of the SCAM. Channel attention is realized by residual
    group, spatial attention is realized by SpatialAttention module.
    Based on the thesis, this module has only one Residual group.
    """
    def __init__(self, conv, n_feats, kernel_size, reduction, act,
                 res_scale, n_resblocks, n_feats_sa_mid):
        super().__init__()

        modules = []
        modules.append(rcan.ResidualGroup(conv, n_feats, kernel_size, reduction, act,
                                          res_scale, n_resblocks))
        modules.append(SpatialAttention(n_feats, n_feats_sa_mid, act))

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.tensor(B,C,H,W)): Feature map.

        Returns:
            torch.tensor(B,1,H,W): Spatial attention map.
        """
        x = self.body(x)
        return x


class SCAM(nn.Module):
    """Spatial Color Attention Module (SCAM).
    
    This module implements attention branch of the SCAN. This module
    supports only 1, 3, and 4 color dimensions.

    Todo:
        - Hack: Expand restricted color channel dimension.
    """
    def __init__(self, conv, n_feats, kernel_size, reduction, act,
                 res_scale, n_resblocks, n_feats_sa_mid, n_colors):
        super().__init__()

        if n_colors not in (1, 3, 4):
            raise ValueError('n_colors must be 1/3/4, but {}'.format(n_colors))

        self.n_colors = n_colors

        self.red = nn.Sequential(*[
            nn.Conv2d(1, n_feats, kernel_size, padding=kernel_size//2, bias=True),
            RCSA(conv, n_feats, kernel_size, reduction, act, res_scale, n_resblocks, n_feats_sa_mid),
            ])

        if n_colors >= 3:
            self.green = nn.Sequential(*[
                nn.Conv2d(1, n_feats, kernel_size, padding=kernel_size//2, bias=True),
                RCSA(conv, n_feats, kernel_size, reduction, act, res_scale, n_resblocks, n_feats_sa_mid),
                ])
            self.blue = nn.Sequential(*[
                nn.Conv2d(1, n_feats, kernel_size, padding=kernel_size//2, bias=True),
                RCSA(conv, n_feats, kernel_size, reduction, act, res_scale, n_resblocks, n_feats_sa_mid),
                ])

        if n_colors >= 4:
            self.alpha = nn.Sequential(*[
                nn.Conv2d(1, n_feats, kernel_size, padding=kernel_size//2, bias=True),
                RCSA(conv, n_feats, kernel_size, reduction, act, res_scale, n_resblocks, n_feats_sa_mid),
                ])

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.tensor(B,n_colors,H,W)): Original input images,
                where n_colors is dimension of color channel.

        Returns:
            Tuple of torch.tensor(B,n_colors,H,W): Spatial attention
                masks for each color channel calculated separatly.

        Note:
            clone(): Copy data without sharing memory.
            detach(): Copy data without reqires_grad(= auto gradient
                calculation)
            clone().detach(): Useful when input should be regarded as
                constant input.
        """
        # (B,n_colors,H,W) -> (B,H,W) -> (B,C,H,W)
        x_r = x[:, 0, :, :].unsqueeze(1).clone().detach()
        # (B,C,H,W) -> (B,C,H,W)
        x_r = self.red(x_r)

        if self.n_colors < 3:
            return (x_r,)

        x_g = x[:, 1, :, :].unsqueeze(1).clone().detach()
        x_g = self.green(x_g)
        x_b = x[:, 2, :, :].unsqueeze(1).clone().detach()
        x_b = self.blue(x_b)

        if self.n_colors < 4:
            return (x_r, x_g, x_b)

        x_a = x[:, 3, :, :].unsqueeze(1).clone().detach()
        x_a = self.alpha(x_a)

        return (x_r, x_g, x_b, x_a)  # 4*(B,C,H,W)


class BaseNet(nn.Module):
    """Base network for SCAN.

    This module implements base network of the SCAN. SCAN repeatly
    multiply spatial attention mask to feature map.

    Todo:
        - Three layers should be re-written using for statement.
    """
    def __init__(self, conv, n_feats, kernel_size, reduction, act,
                 res_scale, n_resblocks, n_resgroups, n_colors):
        super().__init__()

        self.n_colors = n_colors

        self.RGs1 = nn.Sequential(*[
            rcan.ResidualGroup(conv, n_feats, kernel_size, reduction, act, res_scale, n_resblocks) \
            for _ in range(n_resgroups // 3)
            ])
        self.RGs2 = nn.Sequential(*[
            rcan.ResidualGroup(conv, n_feats, kernel_size, reduction, act, res_scale, n_resblocks) \
            for _ in range(n_resgroups // 3)
            ])
        self.RGs3 = nn.Sequential(*[
            rcan.ResidualGroup(conv, n_feats, kernel_size, reduction, act, res_scale, n_resblocks) \
            for _ in range(n_resgroups // 3)
            ])

        self.conv1 = nn.Conv2d(n_feats * n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        self.conv2 = nn.Conv2d(n_feats * n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)
        self.conv3 = nn.Conv2d(n_feats * n_colors, n_feats, kernel_size, padding=kernel_size//2, bias=True)

    def forward(self, x, attn=None):
        """Forward function.

        Args:
            x (torch.tensor(B,C,H,W)): Feature map.
            attn (tuple of torch.tensor(B,1,H,W), optional):
                Spatial attention mask. Defaults to None.

        Returns:
            torch.tensor(B,C,H,W): Feature map.
        """
        if attn is not None and len(attn) != self.n_colors:
            raise RuntimeError('length of attnetion mask list: {} must be same as n_colors: {}'
                               .format(len(attn), self.n_colors))

        # layer 1
        res = self.RGs1(x)  # (B,C,H,W)
        x = x + res         # (B,C,H,W)

        if attn is not None:
            # (B,C,H,W) -> (B,n_colors*C,H,W)
            x = self.mul_spatial_attention(x, attn)
            x = self.conv1(x)  # (B,n_colors*C,H,W) -> (B,C,H,W)

        # layer 2
        res = self.RGs2(x)
        x = x + res

        if attn is not None:
            x = self.mul_spatial_attention(x, attn)
            x = self.conv2(x)

        # layer 3
        res = self.RGs3(x)
        x = x + res

        if attn is not None:
            x = self.mul_spatial_attention(x, attn)
            x = self.conv3(x)

        return x

    def mul_spatial_attention(self, x, attn):
        """Multiply spatial attention mask to feature map.

        Args:
            x (torch.tensor(B,C,H,W)): Feature map.
            attn (tuple of torch.tensor(B,1,H,W)):
                Tuple of attention masks (n_colors, (B,1,H,W)).

        Returns:
            torch.tensor(B,n_colors*C,H,W): Weighted feature map.
        """
        x_mul = []
        for i in range(self.n_colors):
            x_i = x.clone().detach()  # (B,C,H,W)
            x_i *= attn[i]            # (B,C,H,W) * (B,C,H,W) -> (B,C,H,W)
            x_mul.append(x_i)
        x = torch.cat(x_mul, 1)  # n_colors*(B,C,H,W) -> (B,n_colors*C,H,W)

        return x # (B,n_colors*C,H,W)


class SCAN(nn.Module):
    """Spatial Color Attention Network (SCAN).

    This network is advanced network of RCAN. By adding new spatial
    attention branch for each color channel separatly, SCAN overcomes
    RCAN.
    """
    def __init__(self, scale=4, base_n_resblocks=20, base_n_feats=64, base_n_resgroups=9,
                 branch_n_resblocks=6, branch_n_feats=64, branch_n_feats_sa_mid=64,
                 n_colors=3, n_out_colors=3,
                 act=nn.ReLU(inplace=True)
                 ):

        super().__init__()

        self.n_colors = n_colors

        reduction = 16
        res_scale = 1
        kernel_size = 3
        conv = rcan.default_conv

        # define attention branch module
        self.branch = SCAM(conv, branch_n_feats, kernel_size, reduction, act,
                           res_scale, branch_n_resblocks, branch_n_feats_sa_mid, n_colors)

        # define head module
        self.head = conv(n_colors, base_n_feats, kernel_size)

        # define body modules
        self.body = BaseNet(conv, base_n_feats, kernel_size, reduction, act,
                            res_scale, base_n_resblocks, base_n_resgroups, n_colors)

        # define tail module
        self.upscale = rcan.Upsampler(conv, scale, base_n_feats)
        self.tail = conv(base_n_feats, n_out_colors, kernel_size)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.tensor(B,n_colors,H,W)): Original input images.

        Returns:
            torch.tensor(B,n_out_colors,rH,rW): Super resolution images.
        """
        attn = self.branch(x)     # (B,n_colors,H,W) -> n_colors*(B,C,H,W)

        x = self.head(x)          # (B,n_colors,H,W) -> (B,C,H,W)

        res = self.body(x, attn)  # (B,C,H,W), n_colors*(B,C,H,W) -> (B,C,H,W)
        res += x                  # (B,C,H,W) -> (B,C,H,W)

        x = self.upscale(res)     # (B,C,H,W) -> (B,C,rH,rW)
        x = self.tail(x)          # (B,C,rH,rW) -> (B,n_out_colors,rH,rW)

        return x

    def load_state_dict(self, state_dict, strict=False):
        """Load weight data from the file.

        Args:
            state_dict (dict): a dict containing parameters of a model.
            strict (bool, optional): [description]. Defaults to False.

        Examples:
            Generate model itself before load weight.
            >>> model = SCAN()
            Then, load from saved file.
            >>> model.load_state_dict(torch.load(model_path, \
            map_location=torch.device('cpu')), strict=True)
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'
                            .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))


if __name__ == '__main__':
    model = SCAN()