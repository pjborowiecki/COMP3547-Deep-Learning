import torch
import torch.nn as nn


class Residual(nn.Module):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_channels,
        groups_number, 
        dropout_rate=0.1
    ):
        super().__init__()

        self.normalisation_1 = nn.GroupNorm(groups_number, in_channels)
        self.convolution_1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.normalisation_2 = nn.GroupNorm(groups_number, out_channels)
        self.convolution_2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.skip = nn.Identity()

        self.time_embedding = nn.Linear(time_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
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

        y = self.convolution_1(self.swish_activation(self.normalisation_1(x)))
        y += self.time_embedding(self.swish_activation(t))[:, :, None, None]
        y = self.convolution_2(self.dropout(self.swish_activation(self.normalisation_2(y))))
        return y + self.skip(x)