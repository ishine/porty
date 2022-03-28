import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EmbeddingLayer
from .transformer import Transformer
from .predictors import VarianceAdopter
from .flow import Flow
from .posterior_encoder import PosteriorEncoder
from .signal_generator import SignalGenerator
from .gan import Generator
from .monotonic_align import maximum_path

from .loss import kl_loss
from .utils import sequence_mask, rand_slice_segments


class VITS(nn.Module):
    def __init__(self, params):
        super(VITS, self).__init__()
        self.segment_size = params.mel_segment

        self.emb = EmbeddingLayer(**params.embedding)
        self.encoder = Transformer(**params.encoder)
        self.va = VarianceAdopter(**params.va)
        self.stat_proj = nn.Conv1d(params.encoder.channels, params.encoder.channels * 2, 1)

        self.flow = Flow(**params.flow)
        self.posterior_encoder = PosteriorEncoder(**params.posterior_encoder)
        self.generator = Generator(**params.generator)

    def forward(self, inputs):
        phoneme, is_accent, x_length = inputs
        x = self.emb(phoneme)
        is_accent = is_accent.unsqueeze(1)
        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)

        x = self.encoder(x, x_mask)
        x, y_mask, preds = self.va.infer(x, is_accent, x_mask)
        x = self.decoder(x, y_mask)
        x = self.stat_proj(x) * y_mask

        m, logs = torch.chunk(x, 2, dim=1)
        z_p = (m + torch.randn_like(m) * torch.exp(logs)) * y_mask
        z = self.flow.backward(z_p, y_mask)
        y = self.generator(z)
        return y, preds

    def compute_loss(self, batch):
        (
            phoneme, _,
            x_length,
            _,
            spec,
            _,
            y_length,
            *_
        ) = batch
        x = self.emb(phoneme)

        x_mask = sequence_mask(x_length).unsqueeze(1).to(x.dtype)
        y_mask = sequence_mask(y_length).unsqueeze(1).to(x.dtype)

        x = self.encoder(x, x_mask)
        stats = self.stat_proj(x) * x_mask
        m_p, logs_p = torch.chunk(stats, 2, dim=1)

        z, m_q, logs_q = self.posterior_encoder(spec, y_mask)
        z_p = self.flow(z, y_mask)

        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        with torch.no_grad():
            x_s_sq_r = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z_p ** 2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((m_p * x_s_sq_r).transpose(1, 2), z_p)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (m_p ** 2) * x_s_sq_r, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']

            path = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        m_p, logs_p, dur_pred = self.va(
            x,
            m_p,
            logs_p,
            x_mask,
            path.squeeze(1)
        )
        duration = path.sum(dim=-1).add(1e-5).log()

        _kl_loss = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)
        duration_loss = (dur_pred - duration).pow(2).sum() / torch.sum(x_length)
        loss = _kl_loss + duration_loss

        loss_dict = dict(
            loss=loss,
            kl_loss=_kl_loss,
            duration=duration_loss
        )

        z_slice, ids_slice = rand_slice_segments(z_p, y_length, self.segment_size)
        o = self.generator(z_slice)

        return o, ids_slice, loss_dict

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True, remove_wn: bool = True):
        super(VITS, self).load_state_dict(state_dict, strict)
        if remove_wn:
            self.va.remove_weight_norm()
            self.flow.remove_weight_norm()
            self.posterior_encoder.remove_weight_norm()
            self.generator.remove_weight_norm()
