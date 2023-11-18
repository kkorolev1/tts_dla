import numpy as np
from pathlib import Path
import torch
import torchaudio
from tqdm.auto import tqdm
import pyworld as pw
from scipy.interpolate import interp1d


def setup_energy(data_dir):
    """
    Setup energy from LJSpeech mel spectrograms
    """
    save_dir = data_dir / "data" / "energy"
    mel_dir = data_dir / "data" / "mels"
    save_dir.mkdir(exist_ok=True, parents=True)

    assert mel_dir.exists(), "Mel dir not found, download data first"

    min_energy = 1e10
    max_energy = -1e10

    for fpath in mel_dir.iterdir():
        mel = np.load(fpath)
        # energy is L2-norm of mel spec rows
        energy = np.linalg.norm(mel, axis=-1)
        new_name = fpath.name.replace("mel", "energy")
        np.save(save_dir / new_name, energy)     
        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())

    print("min energy:", min_energy, "max energy:", max_energy)


def setup_pitch(data_dir):
    """
    Setup pitch from LJSpeech wavs and mels
    """
    wav_dir = data_dir / "data" / "LJSpeech-1.1" / "wavs"
    mel_dir = data_dir / "data" / "mels"
    save_dir = data_dir / "data" / "pitch"
    save_dir.mkdir(exist_ok=True, parents=True)

    assert wav_dir.exists(), "Wav dir not found, download data first"
    
    names = []
    for fpath in wav_dir.iterdir():
        names.append(fpath.name)

    names_dict = {name: i for i, name in enumerate(sorted(names))}

    min_pitch = 1e10
    max_pitch = 1e-10

    for fpath in tqdm(wav_dir.iterdir(), total=len(names)):
        real_i = names_dict[fpath.name]
        new_name = "ljspeech-pitch-%05d.npy" % (real_i+1)
        mel_name = "ljspeech-mel-%05d.npy" % (real_i+1)

        mel = np.load(mel_dir / mel_name)

        audio, sr = torchaudio.load(fpath)
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

        new_f0 = f(np.arange(f0.shape[0]))

        np.save(save_dir / new_name, new_f0)

        min_pitch = min(min_pitch, new_f0.min())
        max_pitch = max(max_pitch, new_f0.max())

    print("min pitch:", min_pitch, "max pitch:", max_pitch)

if __name__ == "__main__":
    data_dir = Path(".")
    
    print("Setup energy")
    setup_energy(data_dir)

    print("Setup pitch")
    setup_pitch(data_dir)