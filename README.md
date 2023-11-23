# TTS HW 3

Implementation of a TTS pipeline using Fastspeech2 model trained on a LJSpeech dataset.

[WanDB Report](https://wandb.ai/kkorolev/tts_project/reports/HW3-TTS--Vmlldzo2MDQ1MTg5)

## Installation guide

```shell
pip install -r ./requirements.txt
```

To reproduce training download necessary files, including LJSpeech, it's mel spectrograms, alignments for FastSpeech and Waveglow model weights, using a shell script. 
```shell
sh scripts/download_data.sh
```
To get pitches and energies run scripts/setup.py, which saves them in the same data folder as other files.
```shell
python scripts/setup.py
```

Configs can be found in hw_tts/configs folder. In particular, for testing use `config_test.json`.

## Training
One can redefine parameters which are set within config by passing them in terminal via flags.
```shell
python train.py -c CONFIG -r CHECKPOINT -k WANDB_KEY --wandb_run_name WANDB_RUN_NAME --n_gpu NUM_GPU --batch_size REAL_BATCH_SIZE --batch_expand_size MULTIPLIER_FOR_BATCH_SAMPLING --len_epoch ITERS_PER_EPOCH --waveglow_path WAVEGLOW_WEIGHTS_PATH --data_path PATH_TO_TRAIN_TEXTS --mel_ground_truth PATH_TO_GT_MELS --alignment_path PATH_TO_GT_ALIGNMENTS --pitch_path PATH_TO_GT_PITCHES --energy_path PATH_TO_GT_ENERGIES
```

## Testing
To use model with or without prelayer norm, add `"attn_use_prelayer_norm": true/false` to model's config.
```shell
python test.py -c hw_tts/configs/config_test.json -r CHECKPOINT -t test.txt -o output
```
- `test.txt` is a file with 3 sentences for evaluation, each on a newline.
- `output` is a folder to save the result.

You can tune parameters for speeding up / slowing down, pitching up or down, changing energy of an audio. One can find variants of parameters, called `params_list`, with which the audio will be generated also in `config_test.json`. They are given as a list of triplets, where first one is related to duration (greater value means slowing down), second one to pitch (greater value means pitching up), and the third one to energy (greater means lower volume).


Current parameters list generates the following audios for texts given in `test.txt`:
- regular generated audio
- audio with +20%/-20% for pitch/speed/energy
- audio with +20/-20% for pitch, speed and energy together
