import torch
import torch.nn as nn

from .layers import WN


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super().__init__()

        self.pre = nn.Conv1d(in_channels, out_channels, 1)
        self.enc = WN(out_channels, kernel_size, dilation_rate, n_layers)
        self.proj = nn.Conv1d(out_channels, out_channels * 2, 1)

    def forward(self, x, x_mask):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.chunk(stats, chunks=2, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
