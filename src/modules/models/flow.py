import torch
import torch.nn as nn

from .layers import WN


class Flow(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=6):
        super().__init__()

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, kernel_size, dilation_rate, n_layers))
            self.flows.append(Flip())

    def forward(self, x, x_mask):
        for flow in self.flows:
            x = flow(x, x_mask)

    def backward(self, x, x_mask):
        for flow in reversed(self.flows):
            x = flow.backward(x, x_mask)
        return x

    def remove_weight_norm(self):
        for flow in self.flows:
            flow.remove_weight_norm()


class Flip(nn.Module):
    def forward(self, x, x_mask):
        print(x)
        x = torch.flip(x, dims=[1])
        return x

    def backward(self, x, x_mask):
        return x

    def remove_weight_norm(self):
        pass


class ResidualCouplingLayer(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2

        self.pre = nn.Conv1d(channels // 2, channels, 1)
        self.enc = WN(channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout)
        self.post = nn.Conv1d(channels, channels // 2, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask):
        x0, x1 = torch.chunk(x, chunks=2, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask)
        stats = self.post(h) * x_mask

        m = stats
        logs = torch.zeros_like(m)

        x1 = m + x1 * torch.exp(logs) * x_mask
        x = torch.cat([x0, x1], dim=1)
        return x,

    def backward(self, x, x_mask):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        m = stats
        logs = torch.zeros_like(m)
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        x = torch.cat([x0, x1], dim=1)
        return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
