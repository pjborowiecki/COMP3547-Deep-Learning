import torch
import torch.nn as nn

import attention
import residual


class DownBlock(nn.Module):
    
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_channels, 
        has_attention,
        groups_number,
        heads_number
    ):  
        super().__init__()
        self.res = residual.Residual(
            in_channels, 
            out_channels, 
            time_channels, 
            groups_number
        )
        
        if has_attention:
            self.attention = attention.Attention(
                out_channels, 
                groups_number, 
                heads_number
            )
        else:
            self.attention = nn.Identity()

    def forward(
        self, 
        x, 
        t
    ):
        x = self.res(x, t)
        x = self.attention(x)
        return x


class UpBlock(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_channels, 
        has_attention,
        groups_number,
        heads_number
    ):
        super().__init__()
        self.res = residual.Residual(
            in_channels + out_channels, 
            out_channels, 
            time_channels, 
            groups_number
        )
        
        if has_attention:
            self.attention = attention.Attention(
                out_channels, 
                groups_number, 
                heads_number
            )
        else:
            self.attention = nn.Identity()

    def forward(
        self, 
        x, 
        t
    ):
        x = self.res(x, t)
        x = self.attention(x)
        return x


class Downsample(nn.Module):
    
    def __init__(
        self, 
        feature_map_size
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            feature_map_size, 
            feature_map_size, 
            (3, 3), 
            (2, 2), 
            (1, 1)
        )

    def forward(
        self, 
        x, 
        t
    ):

        return self.conv(x)


class Upsample(nn.Module):

    def __init__(
        self, 
        feature_map_size
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            feature_map_size, 
            feature_map_size, 
            (4, 4), 
            (2, 2), 
            (1, 1)
        )

    def forward(
        self, 
        x, 
        t
    ):

        return self.conv(x)
    

class Bottleneck(nn.Module):

    def __init__(
        self, 
        feature_map_size, 
        time_channels,
        groups_number,
        heads_number
    ):
        super().__init__()
        
        self.res1 = residual.Residual(feature_map_size, feature_map_size, time_channels, groups_number)
        self.attn = attention.Attention(feature_map_size, groups_number, heads_number)
        self.res2 = residual.Residual(feature_map_size, feature_map_size, time_channels, groups_number)

    def forward(
        self, 
        x, 
        t
    ):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x