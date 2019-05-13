# audio_tagging

## extract features
set the right paths for audio files in the config files first
* python main.py -c config/features.ini
* python main.py -c config/labels.ini

## train a modell
after feature extraction
* python main.py -c config/train.ini

## Requirements
- muda package for data augmentation (pip install muda)
