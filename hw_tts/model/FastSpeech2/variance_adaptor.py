import torch
import torch.nn as nn
import numpy as np

from hw_tts.model.FastSpeech2.length_regulator import LengthRegulator
from hw_tts.model.FastSpeech2.scalar_predictor import ScalarPredictor


class VarianceAdaptor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.length_regulator = LengthRegulator(**kwargs)

        pitch_space = torch.linspace(np.log(
            kwargs["min_pitch"] + 1), np.log(kwargs["max_pitch"] + 2), kwargs["num_bins"])
        self.register_buffer('pitch_space', pitch_space)
        self.pitch_predictor = ScalarPredictor(
            embed_dim=kwargs["encoder_dim"],
            predictor_filter_size=kwargs["pitch_predictor_filter_size"],
            predictor_kernel_size=kwargs["pitch_predictor_kernel_size"],
            dropout=kwargs["dropout"]
        )
        self.pitch_embedding = nn.Embedding(
            kwargs["num_bins"], kwargs["encoder_dim"])

        energy_space = torch.linspace(np.log(
            kwargs["min_energy"] + 1), np.log(kwargs["max_energy"] + 2), kwargs["num_bins"])
        self.register_buffer('energy_space', energy_space)
        self.energy_predictor = ScalarPredictor(
            embed_dim=kwargs["encoder_dim"],
            predictor_filter_size=kwargs["energy_predictor_filter_size"],
            predictor_kernel_size=kwargs["energy_predictor_kernel_size"],
            dropout=kwargs["dropout"]
        )
        self.energy_embedding = nn.Embedding(
            kwargs["num_bins"], kwargs["encoder_dim"])

    def energy(self, len_reg_output, energy_target=None, gamma=1.0):
        energy_pred = self.energy_predictor(len_reg_output)
        if energy_target is not None:
            buckets = torch.bucketize(
                torch.log(energy_target + 1), self.energy_space)
        else:
            estimated_energy = torch.exp(
                energy_pred) - 1 
            estimated_energy = estimated_energy * gamma
            buckets = torch.clip(torch.bucketize(
                torch.log(estimated_energy + 1), self.energy_space), 0, self.energy_embedding.num_embeddings - 1)
        energy_emb = self.energy_embedding(buckets)
        return energy_emb, energy_pred

    def pitch(self, len_reg_output, pitch_target=None, beta=1.0):
        pitch_pred = self.pitch_predictor(len_reg_output)
        if pitch_target is not None:
            buckets = torch.bucketize(
                torch.log(pitch_target + 1), self.pitch_space)
        else:
            estimated_pitch = torch.exp(
                pitch_pred) - 1
            estimated_pitch = estimated_pitch * beta 
            buckets = torch.clip(torch.bucketize(torch.log(
                estimated_pitch + 1), self.pitch_space), 0, self.pitch_embedding.num_embeddings - 1)
        pitch_emb = self.pitch_embedding(buckets)
        return pitch_emb, pitch_pred

    def forward(self, enc_output, mel_max_length=None,
                length_target=None, pitch_target=None, energy_target=None,
                alpha=1.0, beta=1.0, gamma=1.0):
        if self.training:
            len_reg_output, duration_pred = self.length_regulator(
                enc_output, alpha=alpha, target=length_target, mel_max_length=mel_max_length
            )
            pitch_emb, pitch_pred = self.pitch(
                len_reg_output, pitch_target, beta=beta
            )
            energy_emb, energy_pred = self.energy(
                len_reg_output, energy_target, gamma=gamma
            )
            output = len_reg_output + pitch_emb + energy_emb
            return {
                "output": output,
                "duration_predictor_output": duration_pred,
                "pitch_predictor_output": pitch_pred,
                "energy_predictor_output": energy_pred,
            }
        len_reg_output, mel_pos = self.length_regulator(
            enc_output, alpha=alpha
        )
        pitch_emb = self.pitch(len_reg_output, beta=beta)[0]
        energy_emb = self.energy(len_reg_output, gamma=gamma)[0]
        output = len_reg_output + pitch_emb + energy_emb
        return {
            "output": output,
            "mel_pos": mel_pos
        }
