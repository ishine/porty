import torch
import torch.nn as nn


class SignalGenerator(nn.Module):
    def __init__(self, stats_dir, sr, frame_shift, alpha=0.1, sigma=0.003):
        super(SignalGenerator, self).__init__()
        self.stats = torch.load(f'{stats_dir}/stats.pt')
        self.sr = sr
        self.frame_shift = frame_shift
        self.alpha = alpha
        self.sigma = sigma

    def restore(self, f0):
        return torch.exp(f0 * self.stats['pitch_std'] + self.stats['pitch_mean']).float()

    @torch.no_grad()
    def forward(self, f0, vuv):
        f0 = self.restore(f0)
        f0 *= vuv
        f0 = torch.repeat_interleave(f0, self.frame_shift, dim=-1)
        vuv = torch.repeat_interleave(vuv, self.frame_shift, dim=-1)
        signal = self._signal(f0, vuv)
        return signal

    def _signal(self, f0, vuv):
        noise = torch.randn_like(f0) * self.sigma
        e_v = self.alpha * torch.sin(torch.cumsum(2. * torch.pi * f0 / self.sr, dim=-1)) + noise
        e_uv = self.alpha / (3. * self.sigma) * noise
        sig = e_v * vuv + e_uv * (-vuv + 1)
        return sig
