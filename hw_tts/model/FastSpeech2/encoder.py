import torch
import torch.nn as nn

from hw_tts.model.FastSpeech2.fft import FFT
from hw_tts.model.FastSpeech2.utils import get_non_pad_mask, get_attn_key_pad_mask


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        n_position = kwargs["max_seq_len"] + 1

        self.src_word_emb = nn.Embedding(
            kwargs["vocab_size"],
            kwargs["encoder_dim"],
            padding_idx=kwargs["PAD"]
        )

        self.position_enc = nn.Embedding(
            n_position,
            kwargs["encoder_dim"],
            padding_idx=kwargs["PAD"]
        )

        self.layer_stack = nn.ModuleList([FFT(
            embed_dim=kwargs["encoder_dim"],
            hid_dim=kwargs["encoder_conv1d_filter_size"],
            num_heads=kwargs["encoder_head"],
            fft_conv1d_kernel=kwargs["fft_conv1d_kernel"],
            fft_conv1d_padding=kwargs["fft_conv1d_padding"],
            dropout=kwargs["dropout"],
            attn_use_prelayer_norm=kwargs["attn_use_prelayer_norm"]
        ) for _ in range(kwargs["encoder_n_layer"])])

    def forward(self, src_seq, src_pos):
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        
        return enc_output, non_pad_mask


# encoder = Encoder(256, 4, 1000, 5, 512, 500, 0, 0.1)
# dummy = torch.randint(0, 999, (1, 4))
# dummy[0, 2:] = 0
# dummy_pos = torch.rand((1, 4)).int()
# enc_output, non_pad_mask = encoder(dummy, dummy_pos)
# print(enc_output.shape)