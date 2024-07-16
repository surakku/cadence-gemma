from sys import modules
import torch.nn as nn
from torch.utils import checkpoint
from torch import bfloat16


class MLPProjector(nn.Module):
    def __init__(self, device="cuda", hidden_depth=2):
        super().__init__()
        
        self.device = device
        self.hidden_depth = hidden_depth
        self.modules = [
            nn.Linear(in_features=2176, out_features=2560, device=self.device, dtype=bfloat16),
            nn.GELU(),
        ]
        
        for _ in range(0, self.hidden_depth-1):
            self.modules.append(nn.Linear(in_features=2560, out_features=2560, device=self.device, dtype=bfloat16))
            self.modules.append(nn.GELU())

        self.modules.append(nn.Linear(in_features=2560, out_features=2560, device=self.device, dtype=bfloat16))
        
        self.proj = nn.Sequential(*self.modules)
        
    
    def forward(self, x):
        x = checkpoint.checkpoint(
            self.proj,
            x.to(bfloat16)
        )
        return x