import math
import torch
from torch import nn
from torch.nn import functional as F

from .utils import convert_pad_shape
from .layers import LayerNorm


class Transformer(nn.Module):
    def __init__(self, channels, n_heads, kernel_size, dropout, n_layers, window_size=4):
        super(Transformer, self).__init__()
        self.layers = nn.Sequential(*[
            TransformerLayer(
                channels,
                n_heads,
                kernel_size,
                dropout,
                window_size
            ) for _ in range(n_layers)
        ])

    def forward(self, x, x_mask):
        for layer in self.layers:
            x = layer(x, x_mask)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, channels, n_heads, kernel_size, dropout, window_size):
        super(TransformerLayer, self).__init__()
        self.attn = SelfAttention(channels, n_heads, dropout, window_size)
        self.norm1 = LayerNorm(channels)
        self.ffn = FFN(channels, kernel_size, dropout)
        self.norm2 = LayerNorm(channels)

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        y = self.attn(x, attn_mask)
        y = self.drop(y)
        x = self.norm1(x + y)

        y = self.ffn(x)
        y = self.drop(y)
        x = self.norm2(x + y)
        x = x * x_mask
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels, n_heads, dropout=0., window_size=None):
        super().__init__()
        assert channels % n_heads == 0

        self.inter_channels = channels // n_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.inter_channels)

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

        if window_size is not None:
            rel_stddev = self.inter_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(1, self.inter_channels, window_size * 2 + 1) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(1, self.inter_channels, window_size * 2 + 1) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        x = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        B, C, T = query.size()
        query = query.view(B, self.n_heads, self.inter_channels, T)
        key = key.view(B, self.n_heads, self.inter_channels, T)
        value = value.view(B, self.n_heads, self.inter_channels, T)

        scores = torch.matmul(query.transpose(-2, -1) / self.scale, key)
        if self.window_size is not None:
            k_emb = self._get_relative_embeddings(self.emb_rel_k, T)
            rel_logits = torch.matmul(query.transpose(-2, -1) / self.scale, k_emb.unsqueeze(0))
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value.transpose(-2, -1)).transpose(-2, -1)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            v_emb = self._get_relative_embeddings(self.emb_rel_v, T)
            output = output + torch.matmul(relative_weights, v_emb.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)
        output = output.contiguous().view(B, C, T)
        return output

    def _get_relative_embeddings(self, relative_embeddings, length):
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final


class FFN(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super(FFN, self).__init__()

        self.conv_1 = nn.Conv1d(channels, channels * 4, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(channels * 4, channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask
