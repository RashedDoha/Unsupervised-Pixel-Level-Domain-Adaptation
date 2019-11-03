# Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks
A PyTorch implementation of [Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/abs/1603.08155)

## Table of Contents
- [Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks](#unsupervised-pixel-level-domain-adaptation-with-generative-adversarial-networks)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training](#training)
  - [Logging In Tensorboard](#logging-in-tensorboard)
  - [Contribute](#contribute)

## Prerequisites
- Python 3 (tested with Python 3.7)
- PyTorch (tested with torch v1.3 and torchvision v0.4)
- Python packages as specified in [requirements.txt](requirements.txt)


## Installation
```
$ git clone https://github.com/RashedDoha/Unsupervised-Pixel-Level-Domain-Adaptation.git
$ cd Unsupervised-Pixel-Level-Domain-Adaptation/
$ sudo pip3 install -r requirements.txt
```

## Training
Training on CPU
```
python train.py
```
To train on a CUDA enabled GPU, use the --cuda flag: `python train.py --cuda`

## Logging In Tensorboard
To launch tensorboard make sure to set the `--logdir` flag with the same directory as in the command line flag for train.py `--tensorboard_logs`. (default: `logs`)

## Contribute
To contribute to the project, please refer to the [contribution guidelines](CONTRIBUTING.md)


