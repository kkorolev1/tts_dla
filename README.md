# TTS HW 3

Implementation of a TTS pipeline using Fastspeech2 model trained on a LJSpeech dataset.

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

Configs can be found in hw_ss/configs folder. In particular, for testing use config_test.json.

## Training
```shell
python train.py -c CONFIG
```

## Testing
```shell
python test.py -c CONFIG -r CHECKPOINT
```
