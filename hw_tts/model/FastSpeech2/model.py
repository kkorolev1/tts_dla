import torch
import torch.nn as nn
import numpy as np

from hw_tts.base import BaseModel
from hw_tts.model.FastSpeech2.encoder import Encoder
from hw_tts.model.FastSpeech2.decoder import Decoder
from hw_tts.model.FastSpeech2.length_regulator import LengthRegulator
from hw_tts.model.FastSpeech2.scalar_predictor import ScalarPredictor
from hw_tts.model.FastSpeech2.utils import get_mask_from_lengths

class FastSpeech2(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = Encoder(**kwargs)
        self.length_regulator = LengthRegulator(**kwargs)
        
        # we estimane pitch_target + 1, so we add +1 to bounds
        pitch_space = torch.linspace(np.log(kwargs["min_pitch"] + 1), np.log(kwargs["max_pitch"] + 2), kwargs["num_bins"])
        self.register_buffer('pitch_space', pitch_space)

        self.pitch_predictor = ScalarPredictor(
            embed_dim=kwargs["encoder_dim"],
            predictor_filter_size=kwargs["pitch_predictor_filter_size"],
            predictor_kernel_size=kwargs["pitch_predictor_kernel_size"],
            dropout=kwargs["dropout"]
        )
        self.pitch_emb = nn.Embedding(kwargs["num_bins"], kwargs["encoder_dim"])
        
        # we estimane energy_target + 1, so we add +1 to bounds
        energy_space = torch.linspace(np.log(kwargs["min_energy"] + 1), np.log(kwargs["max_energy"] + 2), kwargs["num_bins"])
        self.register_buffer('energy_space', energy_space)

        self.energy_predictor = ScalarPredictor(
            embed_dim=kwargs["encoder_dim"],
            predictor_filter_size=kwargs["energy_predictor_filter_size"],
            predictor_kernel_size=kwargs["energy_predictor_kernel_size"],
            dropout=kwargs["dropout"]
        )
        self.energy_emb = nn.Embedding(kwargs["num_bins"], kwargs["encoder_dim"])

        self.decoder = Decoder(**kwargs)
        self.mel_linear = nn.Linear(kwargs["decoder_dim"], kwargs["num_mels"], bias=False)
    
    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)
    
    def get_pitch(self, x, pitch_target=None, beta=1.0):
        pitch_pred = self.pitch_predictor(x)
        # we estimate pitch_target + 1 to avoid nans
        if pitch_target is not None:
            buckets = torch.bucketize(torch.log(pitch_target + 1), self.pitch_space)
        else:
            estimated_pitch = torch.exp(pitch_pred) - 1 # (pitch_target + 1) - 1
            estimated_pitch = estimated_pitch * beta # pitch_target * beta
            buckets = torch.bucketize(torch.log(estimated_pitch + 1), self.pitch_space)
        emb = self.pitch_emb(buckets)
        return emb, pitch_pred

    def get_energy(self, x, energy_target=None, gamma=1.0):
        energy_predictor_output = self.energy_predictor(x)
        
        # we estimate energy_target + 1 to avoid nans
        if energy_target is not None:
            buckets = torch.bucketize(torch.log(energy_target + 1), self.energy_space)
        else:
            estimated_energy = torch.exp(energy_predictor_output) - 1 # (energy_target + 1) - 1
            estimated_energy = estimated_energy * gamma # energy_target * gamma
            buckets = torch.bucketize(torch.log(estimated_energy + 1), self.energy_space)
        emb = self.energy_emb(buckets)
        return emb, energy_predictor_output
    
    def forward(self, src_seq, src_pos, 
                mel_pos=None, mel_max_length=None,
                length_target=None, pitch_target=None, energy_target=None,
                alpha=1.0, beta=1.0, gamma=1.0, **kwargs):
        enc_output, non_pad_mask = self.encoder(src_seq, src_pos)
        if self.training:
            len_reg_output, duration_predictor_output = self.length_regulator(
                enc_output, alpha=alpha, target=length_target, mel_max_length=mel_max_length
            )
            pitch_emb, pitch_predictor_output = self.get_pitch(len_reg_output, pitch_target, beta=beta)
            energy_emb, energy_predictor_output = self.get_energy(len_reg_output, energy_target, gamma=gamma)
            output = len_reg_output + pitch_emb + energy_emb
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return {
                "mel_output": output,
                "duration_predictor_output": duration_predictor_output,
                "pitch_predictor_output": pitch_predictor_output,
                "energy_predictor_output": energy_predictor_output
            }
        output, mel_pos = self.length_regulator(
            enc_output, alpha=alpha
        )
        pitch_emb, _ = self.get_pitch(output, beta=beta)
        energy_emb, _ = self.get_energy(output, gamma=gamma)
        output = self.decoder(output + pitch_emb + energy_emb, mel_pos)
        output = self.mel_linear(output)
        return {"mel_output": output}
