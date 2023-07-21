import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):

    def __init__(
        self, 
        feature_map_size
    ):
        super().__init__()
        self.feature_map_size = feature_map_size

        self.lin1 = nn.Linear(self.feature_map_size // 4, self.feature_map_size)
        self.lin2 = nn.Linear(self.feature_map_size, self.feature_map_size)
        
    def swish_activation(
        self,
        x
    ):
        return x * torch.sigmoid(x)

    def forward(
        self, 
        t
    ):
        half_dim = self.feature_map_size // 8
        embedding = math.log(10_000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=t.device) * -embedding)
        embedding = t[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=1)
        embedding = self.swish_activation(self.lin1(embedding))
        embedding = self.lin2(embedding)
        return embedding
