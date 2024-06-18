import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, device="cuda:1", hidden_depth=2):
        super().__init__()
        
        self.device = device
        self.hidden_depth = hidden_depth
        self.modules = [
            nn.Linear(in_features=2176, out_features=2560, device=device),
            nn.GELU(),
        ]
        
        for _ in range(0, self.hidden_depth):
            self.modules.append(nn.Linear(in_features=2560, out_features=2560, device=device))
            self.modules.append(nn.GELU())
        
        self.proj = nn.Sequential(*self.modules)
        
        pytorch_total_params = sum(p.numel() for p in self.proj.parameters())
        print(pytorch_total_params)
    
    def forward(self, x):
        return self.proj(x)