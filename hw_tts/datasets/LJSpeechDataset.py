import time
from tqdm.auto import tqdm
import os
import numpy as np
import torch
from hw_tts.text import text_to_sequence


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)
        return txt


def get_data_to_buffer(data_path, mel_ground_truth, pitch_path, energy_path,
                       alignment_path, text_cleaners, batch_expand_size):
    buffer = list()
    text = process_text(data_path)

    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1)
        )
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(
            os.path.join(alignment_path, str(i)+".npy")
        )
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, text_cleaners)
        )
        pitch_gt_name = os.path.join(
            pitch_path, "ljspeech-pitch-%05d.npy" % (i+1)
        )
        pitch_gt_target = np.load(pitch_gt_name).astype(np.float32)
        energy_gt_name = os.path.join(
            energy_path, "ljspeech-energy-%05d.npy" % (i+1)
        )
        energy_gt_target = np.load(energy_gt_name)

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        pitch_gt_target = torch.from_numpy(pitch_gt_target)
        energy_gt_target = torch.from_numpy(energy_gt_target)

        buffer.append({
            "text": character,
            "duration": duration,
            "mel_target": mel_gt_target,
            "pitch": pitch_gt_target,
            "energy": energy_gt_target,
            "batch_expand_size": batch_expand_size
        })

    return buffer


class LJSpeechDataset:
    def __init__(self, data_path, mel_ground_truth,
                 pitch_path, energy_path, alignment_path,
                 text_cleaners, batch_expand_size, limit=None, **kwargs):
        self.buffer = get_data_to_buffer(
            data_path, mel_ground_truth, pitch_path, energy_path, 
            alignment_path, text_cleaners, batch_expand_size
        )

        if limit is not None:
            self.buffer = self.buffer[:limit]

    def __getitem__(self, index):
        return self.buffer[index]

    def __len__(self):
        return len(self.buffer)