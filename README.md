# Combining various Training and Adaptation Algorithms for Ensemble Few-Shot Classification

# Installation
Install packages using pip:
```bash
$ pip install -r requirements.txt
```

# Pre-trained Models
Pre-trained models of training Algorithms for miniImageNet can be found [here](https://drive.google.com/drive/folders/1FwXK1K77qSI6eznJDBcdC9rLIvnFq4yz).

# Dataset Preparation

## miniImageNet: 
Download [miniImageNet.zip](https://drive.google.com/file/d/1QEbHFIOKIM9KmId175QaLK-r22kgd7br/view) and extract it.


## tieredImageNet:
The data we used here is preprocessed by the repo of [FRN](https://github.com/Tsingularity/FRN).

## CUB: 
Download [`CUB.rar`](https://drive.google.com/drive/my-drive) and extract it.

# Training and Testing
The basic configurations are defined in `cfg/`, overwritten by yaml files. 
## Training Algorithms
We give an example for training the 5-way 5-shot PN model with ResNet12 backbone on MiniImageNet in `write-config/write_yaml_PN.py`.

## Testing individual
Before testing, we need to seach for hyperparameters for adaptation algorithms, and we provide an example in `write-config/write_yaml_search.py`. During testing, we provide an example of the individual obtained by combining the PN training algorithm with the finetune adaptation algorithm in `write-config/write_yaml_search.py`.

## Testing ensemble
We give an example for testing the 5-way 5-shot ensemble model with ResNet12 backbone on MiniImageNet. Exemplar running scripts can be found in `train.sh`.


## Acknowlegements

Our implementation is based on the the official implementation of [CloserLookAgainFewShot](https://github.com/Frankluox/CloserLookAgainFewShot)

