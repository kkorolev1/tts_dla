import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_tts.model.FastSpeech2.utils import Transpose
from hw_tts.model.FastSpeech2.attention import MultiheadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, kernel_sizes, paddings, dropout=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(
                d_in, d_hid, kernel_size=kernel_sizes[0], padding=paddings[0]),
            nn.ReLU(),
            nn.Conv1d(
                d_hid, d_in, kernel_size=kernel_sizes[1], padding=paddings[1]),
            Transpose(1, 2),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, x):
        return self.layer_norm(self.sequential(x) + x)


class FFT(nn.Module):
    def __init__(self, embed_dim, hid_dim, num_heads,
                 fft_conv1d_kernel, fft_conv1d_padding, dropout=0.1, attn_use_prelayer_norm=True):
        super().__init__()
        self.slf_attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout,
            attn_use_prelayer_norm=attn_use_prelayer_norm
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_in=embed_dim, d_hid=hid_dim,
            kernel_sizes=fft_conv1d_kernel,
            paddings=fft_conv1d_padding,
            dropout=dropout
        )

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output = self.slf_attn(
            enc_input, mask=slf_attn_mask, return_attention=False
        )

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)

        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output


# hidden_size = 16
# intermediate_size = 64
# n_head = 4
# batch_size = 4
# seq_len = 12

# fft_block = FFT(hidden_size, intermediate_size, n_head)
# inp_tensor = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.float32)
# out_tensor = fft_block(inp_tensor)
# assert inp_tensor.shape == out_tensor.shape
