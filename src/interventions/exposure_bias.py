import torch.nn as nn


class ExposureBias(nn.Module):
    def __init__(self, exp_dim=4, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(exp_dim),
            nn.Linear(exp_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, exposure):
        return self.net(exposure).squeeze(-1)
