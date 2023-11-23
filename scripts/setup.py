import numpy as np
from pathlib import Path
import torch
import torchaudio
from tqdm.auto import tqdm
import os
import pyworld as pw
from scipy.interpolate import interp1d


def setup_energy(data_dir):
    """
    Setup energy from LJSpeech mel spectrograms
    """
    mels_dir = os.path.join(data_dir, "mels")
    output_dir = os.path.join(data_dir, "energy")
    os.makedirs(output_dir, exist_ok=True)

    min_energy = float("inf")
    max_energy = float("-inf")

    for mel_filename in tqdm(os.listdir(mels_dir), desc="Processing energy"):
        # energy is L2-norm of mel spec columns
        energy = np.linalg.norm(
            np.load(os.path.join(mels_dir, mel_filename)), axis=-1)
        energy_filename = mel_filename.replace("mel", "energy")
        np.save(os.path.join(output_dir, energy_filename), energy)
        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())

    print("min energy:", min_energy)
    print("max energy:", max_energy)


def setup_pitch(data_dir):
    """
    Setup pitch from LJSpeech wavs and mels
    """
    mels_dir = os.path.join(data_dir, "mels")
    wavs_dir = os.path.join(data_dir, "LJSpeech-1.1", "wavs")
    output_dir = os.path.join(data_dir, "pitch")
    os.makedirs(output_dir, exist_ok=True)

    filename_to_id = {filename: i for i, filename in enumerate(
        sorted(os.listdir(wavs_dir)), 1)}

    min_pitch = float("inf")
    max_pitch = float("-inf")

    for wav_filename in tqdm(sorted(os.listdir(wavs_dir)[:1]), desc="Processing pitch"):
        id = filename_to_id[wav_filename]
        pitch_filename = "ljspeech-pitch-{:05d}.npy".format(id)
        mel_filename = "ljspeech-mel-{:05d}.npy".format(id)

        mel = np.load(os.path.join(mels_dir, mel_filename))

        audio, sr = torchaudio.load(os.path.join(wavs_dir, wav_filename))
        audio = audio.to(torch.float64).numpy().sum(axis=0)

        # from audio Hz to mel hz
        frame_period = (audio.shape[0] / sr * 1000) / mel.shape[0]
        # raw pitch extractor
        _f0, t = pw.dio(audio, sr, frame_period=frame_period)
        # pitch refinement
        f0 = pw.stonemask(audio, _f0, t, sr)[:mel.shape[0]]

        # interpolate zero values with values on the bounds
        nonzeros = np.nonzero(f0)
        x = np.arange(f0.shape[0])[nonzeros]

        values = (f0[nonzeros][0], f0[nonzeros][-1])
        f = interp1d(x, f0[nonzeros], bounds_error=False, fill_value=values)

        pitch_contour = f(np.arange(f0.shape[0]))

        np.save(os.path.join(output_dir, pitch_filename), pitch_contour)

        min_pitch = min(min_pitch, pitch_contour.min())
        max_pitch = max(max_pitch, pitch_contour.max())

    print("min pitch:", min_pitch)
    print("max pitch:", max_pitch)


if __name__ == "__main__":
    data_dir = "data"

    print("Setup energy")
    setup_energy(data_dir)

    print("Setup pitch")
    setup_pitch(data_dir)
