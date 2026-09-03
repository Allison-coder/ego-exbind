import torch.nn as nn
import torch.nn.functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, residual_scale=1.0, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.residual_scale = float(residual_scale)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h):
        a = self.fc2(self.drop(self.act(self.fc1(self.ln(h)))))
        return F.normalize(h + self.residual_scale * a, dim=-1)


class DualResidualAdapters(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64, residual_scale=1.0, dropout=0.0):
        super().__init__()
        self.video_adapter = ResidualAdapter(input_dim, hidden_dim, residual_scale, dropout)
        self.text_adapter = ResidualAdapter(input_dim, hidden_dim, residual_scale, dropout)

    def forward(self, video_h, text_h):
        return self.video_adapter(video_h), self.text_adapter(text_h)
