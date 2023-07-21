import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(
        self, 
        feature_map_size, 
        groups_number,
        heads_number,
    ):
        super().__init__()
        self.head_size = feature_map_size
        self.heads_number = heads_number
        
        self.norm = nn.GroupNorm(groups_number, feature_map_size)
        self.projection = nn.Linear(feature_map_size, heads_number * self.head_size * 3)
        self.output = nn.Linear(heads_number * self.head_size, feature_map_size)
        self.scale = self.head_size ** -0.5
      
    def forward(
        self, 
        x, 
        t=None
    ):

        batch_size, feature_map_size, height, width = x.shape

        x = x.view(batch_size, feature_map_size, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.heads_number, 3 * self.head_size)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        attention = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attention = attention.softmax(dim=2)

        res = torch.einsum('bijh,bjhd->bihd', attention, v)
        res = res.view(batch_size, -1, self.heads_number * self.head_size)
        res = self.output(res)
        res += x
        res = res.permute(0, 2, 1).view(batch_size, feature_map_size, height, width)
        return res
    

    