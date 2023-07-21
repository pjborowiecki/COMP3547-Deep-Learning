import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    
    def __init__(
        self,
        input_size,
        output_size
    ):
        super().__init__()
        
        self.dense = nn.Linear(input_size, output_size)
        
    def forward(
        self,
        x
    ):
        return self.dense(x)[..., None, None]