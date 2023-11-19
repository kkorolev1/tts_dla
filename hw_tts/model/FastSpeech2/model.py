import torch
import torch.nn as nn
import numpy as np

from hw_tts.base import BaseModel
from hw_tts.model.FastSpeech2.encoder import Encoder
from hw_tts.model.FastSpeech2.decoder import Decoder
from hw_tts.model.FastSpeech2.variance_adaptor import VarianceAdaptor
from hw_tts.model.FastSpeech2.utils import get_mask_from_lengths

class FastSpeech2(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder(**kwargs)
        self.variance_adaptor = VarianceAdaptor(**kwargs)
        self.decoder = Decoder(**kwargs)
        self.mel_linear = nn.Linear(kwargs["decoder_dim"], kwargs["num_mels"])
    
    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)
    
    def forward(self, src_seq, src_pos, 
                mel_pos=None, mel_max_length=None,
                length_target=None, pitch_target=None, energy_target=None,
                alpha=1.0, beta=1.0, gamma=1.0, **kwargs):
        enc_output, non_pad_mask = self.encoder(src_seq, src_pos)
        variance_adaptor_output = self.variance_adaptor(
            enc_output=enc_output,
            mel_max_length=mel_max_length,
            length_target=length_target,
            pitch_target=pitch_target,
            energy_target=energy_target,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        output = variance_adaptor_output["output"]
        if self.training:
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return {
                "mel_output": output,
                "duration_predictor_output": variance_adaptor_output["duration_predictor_output"],
                "pitch_predictor_output": variance_adaptor_output["pitch_predictor_output"],
                "energy_predictor_output": variance_adaptor_output["energy_predictor_output"]
            }
        output = self.decoder(output, variance_adaptor_output["mel_pos"])
        output = self.mel_linear(output)
        return {"mel_output": output}
