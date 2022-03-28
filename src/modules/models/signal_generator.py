import math
import torch
import torch.nn as nn


class SignalGenerator(nn.Module):
    def __init__(self, stats_dir, sr, frame_shift, n_harmonic, alpha=0.1, sigma=0.003):
        super(SignalGenerator, self).__init__()
        stats = torch.load(f'{stats_dir}/stats.pt')
        self.f0_mean = float(stats['pitch_mean'])
        self.f0_std = float(stats['pitch_std'])
        self.sr = sr
        self.frame_shift = frame_shift
        self.n_harmonic = n_harmonic
        self.alpha = alpha
        self.sigma = sigma

        phi = (torch.rand(self.n_harmonic, requires_grad=False) * 2 - 1) * math.pi
        phi[0] = 0
        self.register_buffer('phi', phi)

        self.amplitude = nn.Parameter(torch.ones(self.n_harmonic + 1), requires_grad=True)

    def restore(self, f0):
        return torch.exp(f0 * self.f0_std + self.f0_mean)

    @torch.no_grad()
    def forward(self, f0, vuv):
        f0, vuv = self._preprocess(f0, vuv)
        output = 0
        for i in range(self.n_harmonic):
            output += self.amplitude[i] * self._signal(f0 * (i+1), vuv, self.phi[i])
        output = torch.tanh(output + self.amplitude[self.n_harmonic])
        return output

    @torch.no_grad()
    def _preprocess(self, f0, vuv):
        f0 = self.restore(f0)
        f0 *= vuv
        return f0, vuv

    def _signal(self, f0, vuv, phi):
        noise = torch.randn_like(f0) * self.sigma
        e_v = self.alpha * torch.sin(torch.cumsum(2. * math.pi * f0 / self.sr, dim=-1) + phi) + noise
        e_uv = self.alpha / (3. * self.sigma) * noise
        sig = e_v * vuv + e_uv * (1 - vuv)
        return sig
