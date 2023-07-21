import numpy as np
import torch
import torch.nn as nn



class FeatureProjection(nn.Module):
    
    def __init__(
        self,
        embedding_size,
        scale
    ):
        super().__init__()
        
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)
        
    def forward(
        self,
        x
    ):
        self.W = self.W.to(x.device)
        
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)