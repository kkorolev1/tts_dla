import torch
import torch.nn as nn

from hw_tts.model.FastSpeech2.fft import FFT
from hw_tts.model.FastSpeech2.utils import get_non_pad_mask, get_attn_key_pad_mask


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        n_position = kwargs["max_seq_len"] + 1

        self.position_enc = nn.Embedding(
            n_position,
            kwargs["decoder_dim"],
            padding_idx=kwargs["PAD"]
        )

        self.layer_stack = nn.ModuleList([FFT(
            embed_dim=kwargs["decoder_dim"],
            hid_dim=kwargs["encoder_conv1d_filter_size"],
            num_heads=kwargs["decoder_head"],
            fft_conv1d_kernel=kwargs["fft_conv1d_kernel"],
            fft_conv1d_padding=kwargs["fft_conv1d_padding"],
            dropout=kwargs["dropout"]
        ) for _ in range(kwargs["decoder_n_layer"])])

    def forward(self, enc_seq, enc_pos):
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)
        
        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask
            )
        
        return dec_output
    
# decoder = Decoder(256, 4, 5, 512, 500, 0, 0.1)
# dummy = torch.randint(0, 500, (1, 4, 256))
# dummy[0, 2:] = 0
# dummy_pos = torch.randint(0, 500, (1, 4))
# dec_output = decoder(dummy, dummy_pos)
# print(dec_output.shape)