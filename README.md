# TTS HW 3

Implementation of a TTS pipeline using Fastspeech2 model trained on a LJSpeech dataset.

[WanDB Report](https://wandb.ai/kkorolev/tts_project/reports/HW3-TTS--Vmlldzo2MDQ1MTg5)

See the results of both models at the end of this README.

## Checkpoints
- [First model checkpoint](https://disk.yandex.ru/d/qQx-LW21qd17Xg)
- [Second model checkpoint](https://disk.yandex.ru/d/cW4s6OLS12KyjA)

The first model was trained with the following configuration: batch size=20, batch expand size=24, AdamW with warmup, max_lr=5e-4, len_epoch=3000, num_epochs=40, grad_norm_clip=2.

The second model uses pre layer norm in attention. It was trained with batch size=64, max_lr=1e-3. Initialization in MultiHead Attention was replaced on xavier uniform. All other configuration parameters are the same.
To use model with or without prelayer norm, add `"attn_use_prelayer_norm": true/false` to model's config.

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

## Results
Generation of these 3 sentences. Filename corresponds to the order of a sentence.

`A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`

`Massachusetts Institute of Technology may be best known for its math, science and engineering education`

`Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

I considered to publish results of two models, that were described in a report, because their generation quality difference is quite subjective.
### First model


https://github.com/kkorolev1/tts_dla/assets/72045472/babecb5f-5997-418e-89b7-598d417e2c2a



https://github.com/kkorolev1/tts_dla/assets/72045472/947b5d93-91c4-46a8-83a9-a257f48e7bcd



https://github.com/kkorolev1/tts_dla/assets/72045472/26799178-716f-46a1-abea-eb497e986425



https://github.com/kkorolev1/tts_dla/assets/72045472/281fb6ae-b528-4969-955d-2ae671c37fda



https://github.com/kkorolev1/tts_dla/assets/72045472/acc716a3-72e3-4e1a-9f3f-600d04a3f2c6



https://github.com/kkorolev1/tts_dla/assets/72045472/d88e7c4b-aebb-4152-8bf4-9c571cd5eec7



https://github.com/kkorolev1/tts_dla/assets/72045472/f72f162f-aa70-429f-98dc-a7f1682a7840



https://github.com/kkorolev1/tts_dla/assets/72045472/c0a6c662-8b12-47fb-8f75-16cf143fa715



https://github.com/kkorolev1/tts_dla/assets/72045472/9f4b5e86-1861-4899-be34-26fdc8819c3a



https://github.com/kkorolev1/tts_dla/assets/72045472/9efc3af9-4b1e-425a-a141-58d791e8eda6



https://github.com/kkorolev1/tts_dla/assets/72045472/3cee2220-4e5b-4a4a-b5e4-b93868a05ab8



https://github.com/kkorolev1/tts_dla/assets/72045472/86e0c4cb-4486-4308-ace9-54f95f9c1883



https://github.com/kkorolev1/tts_dla/assets/72045472/910e3195-1d7e-47cf-9ec3-efdb393a3a43



https://github.com/kkorolev1/tts_dla/assets/72045472/9fba43fd-b432-4637-93c6-d234f426ca7c



https://github.com/kkorolev1/tts_dla/assets/72045472/46f16e4c-1d4d-46dc-a490-82a462dc4c29



https://github.com/kkorolev1/tts_dla/assets/72045472/4b8ef82e-a1ea-4ad9-adab-a40474dd2f5b



https://github.com/kkorolev1/tts_dla/assets/72045472/acbe5247-36f7-4f68-94e0-2ef0f4e3a3a7



https://github.com/kkorolev1/tts_dla/assets/72045472/4e3c33e2-3886-4170-aed2-9a37346a171e



https://github.com/kkorolev1/tts_dla/assets/72045472/4934cec3-b147-4894-8d40-a9abea064dd5



https://github.com/kkorolev1/tts_dla/assets/72045472/09f3e62f-2057-4944-904f-b37524337195



https://github.com/kkorolev1/tts_dla/assets/72045472/de4c188a-4898-4639-add0-a607d4dfe23d



https://github.com/kkorolev1/tts_dla/assets/72045472/e92e1552-47c7-41b7-98df-0a11c51b2d5d



https://github.com/kkorolev1/tts_dla/assets/72045472/dc8a6b18-a4c9-46e8-bea3-72c07e111952



https://github.com/kkorolev1/tts_dla/assets/72045472/eb2e2d7e-efc9-4f9c-97a3-e221cf2304e2



https://github.com/kkorolev1/tts_dla/assets/72045472/87f02025-7b72-4fd0-a5d8-da5cabe0db9a



https://github.com/kkorolev1/tts_dla/assets/72045472/19b35bb6-3859-4187-a0ea-63c4bc263d28



https://github.com/kkorolev1/tts_dla/assets/72045472/003be863-69bd-455e-8c10-d0365aceeb85


### Second model



https://github.com/kkorolev1/tts_dla/assets/72045472/c1e63402-ea92-4ff5-94ad-3a411b1159cf



https://github.com/kkorolev1/tts_dla/assets/72045472/827cff10-4b61-49ef-8ab9-a5496c3cb4a7



https://github.com/kkorolev1/tts_dla/assets/72045472/0fc0f66c-604b-4d3b-9c1b-ac25aca573b2



https://github.com/kkorolev1/tts_dla/assets/72045472/7306d46a-c31e-43e2-a916-c2d7ab398eaa



https://github.com/kkorolev1/tts_dla/assets/72045472/4ff3227a-0fd4-4394-8458-cc824eeb6e5d



https://github.com/kkorolev1/tts_dla/assets/72045472/1b6c6c3b-a982-435f-9ea7-7b973116e51b



https://github.com/kkorolev1/tts_dla/assets/72045472/382cabdd-bf93-4179-8ac6-4dbb1d03ffaf



https://github.com/kkorolev1/tts_dla/assets/72045472/c6c1f5ea-102e-44a5-b060-5aa2a949f836



https://github.com/kkorolev1/tts_dla/assets/72045472/313787ca-e1ee-45a4-8a10-9abeeb799cb1



https://github.com/kkorolev1/tts_dla/assets/72045472/4d8e903c-4dde-4e95-85b8-8fdb1a427354



https://github.com/kkorolev1/tts_dla/assets/72045472/802e8cf7-944f-43cf-96df-05e21a118dc8



https://github.com/kkorolev1/tts_dla/assets/72045472/7b304bf4-d6fa-43f9-a02e-71f7a918fb3d



https://github.com/kkorolev1/tts_dla/assets/72045472/fb93501d-77ad-4319-93de-737f7160d1ad



https://github.com/kkorolev1/tts_dla/assets/72045472/a7877020-8ad2-4a4b-9158-c2cbe27e19b7



https://github.com/kkorolev1/tts_dla/assets/72045472/20856bbc-9714-4851-b8c1-32f6da470768



https://github.com/kkorolev1/tts_dla/assets/72045472/c7fd6d92-6f8b-4aee-8b24-a5d424d6c8c7



https://github.com/kkorolev1/tts_dla/assets/72045472/b809b718-d31c-462c-9501-0b14b84d5cc2



https://github.com/kkorolev1/tts_dla/assets/72045472/d65dfb23-ec96-4e4d-ad1d-e6c4928440dc



https://github.com/kkorolev1/tts_dla/assets/72045472/86f40b99-470f-4487-b791-b3a959897990



https://github.com/kkorolev1/tts_dla/assets/72045472/3ad1d909-19be-4b5e-8742-ea8cd55d1b73



https://github.com/kkorolev1/tts_dla/assets/72045472/16612015-e178-4104-accc-6533c653fddc



https://github.com/kkorolev1/tts_dla/assets/72045472/66d0b398-c688-47b4-bfa6-986d5f4c1c87



https://github.com/kkorolev1/tts_dla/assets/72045472/c6c64d86-a0ab-4aab-b077-c6f356b5eb32



https://github.com/kkorolev1/tts_dla/assets/72045472/e4f5bf74-5db8-4c4a-a2da-7244913007f0



https://github.com/kkorolev1/tts_dla/assets/72045472/65530787-2efe-4de0-b479-c5505655d019



https://github.com/kkorolev1/tts_dla/assets/72045472/2986e7b4-1d4a-415b-ad62-a1d84573fc0f



https://github.com/kkorolev1/tts_dla/assets/72045472/5e5b1389-546c-4822-9149-114642cd5016


