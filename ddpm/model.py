import torch
import torch.nn as nn

import time_embedding
import encoder_decoder


class EpsilonTheta(nn.Module):

    def __init__(
        self, 
        channels,
        feature_map_size,
        groups_number,
        heads_number,
        blocks_number
    ):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.groups_number = groups_number
        self.heads_number = heads_number
        self.has_attention = [False, False, False, True]
        
        self.image_proj = nn.Conv2d(channels, feature_map_size, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = time_embedding.TimeEmbedding(feature_map_size * 4)
        
        multipliers = [1, 2, 2, 4]
        n_resolutions = len(multipliers)
        
        # DOWNSAMPLING (ENCODER)
        down = []
        out_channels = in_channels = feature_map_size
        for i in range(n_resolutions):
            out_channels = in_channels * multipliers[i]
            for _ in range(blocks_number):
                down.append(encoder_decoder.DownBlock(
                    in_channels, 
                    out_channels, 
                    feature_map_size * 4, 
                    has_attention=self.has_attention[i], 
                    groups_number=self.groups_number, 
                    heads_number=self.heads_number)
                )
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(encoder_decoder.Downsample(in_channels))
        self.down = nn.ModuleList(down)

        # BOTTLENECK
        self.bottleneck = encoder_decoder.Bottleneck(
            out_channels, 
            feature_map_size * 4, 
            groups_number=self.groups_number, 
            heads_number=self.heads_number,
        )

        # UPSAMPLING (DECODER)
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(blocks_number):
                up.append(encoder_decoder.UpBlock(
                    in_channels, 
                    out_channels, 
                    feature_map_size * 4, 
                    has_attention=self.has_attention[i], 
                    groups_number=self.groups_number,
                    heads_number=self.heads_number)
                )
            out_channels = in_channels // multipliers[i]
            up.append(encoder_decoder.UpBlock(
                in_channels, 
                out_channels, 
                feature_map_size * 4, 
                has_attention=self.has_attention[i], 
                groups_number=self.groups_number,
                heads_number=self.heads_number)
            )
            in_channels = out_channels
            if i > 0:
                up.append(encoder_decoder.Upsample(in_channels))
        self.up = nn.ModuleList(up)
                            
        self.normalisation = nn.GroupNorm(8, feature_map_size)
        self.final = nn.Conv2d(in_channels, channels, kernel_size=(3, 3), padding=(1, 1))
        
    def swish_activation(
        self,
        x
    ):
        return x * torch.sigmoid(x)
        

    def forward(
        self, 
        x, 
        t
    ):

        t = self.time_emb(t)
        x = self.image_proj(x)
                                
        h = [x]
        
        # DOWNSAMPLING LAYER (ENCODER)
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # BOTTLENECK LAYER
        x = self.bottleneck(x, t)

        # UPSAMPLING LAYER (DECODER)
        for m in self.up:
            if isinstance(m, encoder_decoder.Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.swish_activation(self.normalisation(x)))