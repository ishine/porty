import torch
import torch.nn as nn

from .layers import LayerNorm, WN
from .utils import sequence_mask, generate_path


class VarianceAdopter(nn.Module):
    def __init__(self, in_channels, channels, dropout):
        super(VarianceAdopter, self).__init__()
        self.duration_predictor = VariancePredictor(
            in_channels=in_channels,
            channels=channels,
            n_layers=2,
            kernel_size=3,
            dropout=dropout
        )
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = F0Predictor(
            in_channels=in_channels,
            channels=channels,
            n_layers=5,
            kernel_size=5,
            dropout=dropout
        )
        self.energy_predictor = VariancePredictor(
            in_channels=in_channels,
            channels=channels,
            n_layers=2,
            kernel_size=3,
            dropout=dropout
        )

    def forward(
        self,
        x,
        x_mask,
        y_mask,
        pitch,
        energy,
        path
    ):
        dur_pred = torch.relu(self.duration_predictor(x.detach(), x_mask))
        x = self.length_regulator(x, path)
        pitch_pred, vuv = self.pitch_predictor(x, y_mask)
        energy_pred = self.energy_predictor(x, y_mask)

        x = x + pitch + energy
        return x, (dur_pred, pitch_pred, vuv, energy_pred)

    def infer(self, x, is_accent, x_mask):
        dur_pred = torch.relu(self.duration_predictor(x, x_mask))
        accent_mask = (is_accent != 1).float()
        dur_pred = torch.round(torch.exp(dur_pred)) * accent_mask * x_mask
        y_length = torch.clamp_min(torch.sum(dur_pred, [1, 2]), 1).long()
        y_mask = sequence_mask(y_length).unsqueeze(1).to(x_mask.device)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        path = generate_path(dur_pred.squeeze(1), attn_mask.squeeze(1))

        x = self.length_regulator(x, path)

        pitch, vuv = self.pitch_predictor(x, y_mask)
        energy = self.energy_predictor(x, y_mask)

        x = x + pitch + energy
        return x, y_mask, (pitch, vuv, energy)

    def remove_weight_norm(self):
        self.pitch_predictor.remove_weight_norm()


class F0Predictor(nn.Module):
    def __init__(self, in_channels, channels, n_layers, kernel_size, dropout):
        super(F0Predictor, self).__init__()
        self.in_conv = nn.Conv1d(in_channels, channels, 1)
        self.enc = WN(channels, kernel_size, dilation_rate=1, n_layers=n_layers, p_dropout=dropout)
        self.out_conv = nn.Conv1d(channels, 1, 1)
        self.clf = nn.Conv1d(channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.in_conv(x) * x_mask
        x = self.enc(x, x_mask)
        pitch = self.out_conv(x) * x_mask
        vuv = torch.sigmoid(self.clf(x)) * x_mask
        return pitch, vuv

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class VariancePredictor(nn.Module):
    def __init__(self, in_channels, channels, n_layers, kernel_size, dropout):
        super(VariancePredictor, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels if i == 0 else channels, channels, kernel_size, padding=kernel_size // 2),
                LayerNorm(channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for i in range(n_layers)
        ])

        self.out = nn.Conv1d(channels, 1, 1)

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x)
            x *= x_mask
        x = self.out(x)
        x *= x_mask
        return x


class LengthRegulator(nn.Module):
    def forward(self, x, path):
        x = torch.bmm(x, path)
        return x
