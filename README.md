# My Pytorch Program Templete (GNN)

## Requirements

```
torch_scatter==2.0.9
tqdm==4.62.3
torch==1.10.0
numpy==1.20.3
torch_sparse==0.6.12
torch_geometric==2.0.2
PyYAML==6.0
scikit_learn==1.0.2
```

## Usage

The code in this repo is a node classification example on Cora of the template. Try

```
python main.py  --model GCN --dataset Cora
```

to run code.

#### Config file format

Config files are in ``config/{dataset_name}.yml`` format:

```
dataset: "Cora"   

model_name: "GCN"   # Name of the used baseline model, which can be change to 'GAT' of others.

# config for each baseline model
GCN:
  epochs: 150
  multirun: 10
  dropout: 0.5
  cuda: 0
  feat_norm: True
  hidden_dim: 64
  multilabel: False
  patience: 50
  seed: 1234
  lr: 0.005
  weight_decay: 0.0005
  lr_scheduler: False
  monitor: "val_acc"
  recache: False
  optimizer: "Adam"
  num_layers: 2
  activation: "relu"

GAT:
  epochs: 100
  multirun: 10
  dropout: 0.6
  cuda: 0
  feat_norm: True
  hidden_dim: 64
  multilabel: False
  heads: 1
  patience: 50
  seed: 1234
  lr: 0.005
  weight_decay: 0.0005
  lr_scheduler: False
  monitor: "val_acc"
  recache: False
  num_layers: 2
  optimizer: "Adam"
  activation: "leaky_relu"

```

**For each dataset, we need a specific config file. In each file, we config all baseline models.**

#### Model

All baseline models are in ``model/``

#### Trainer

In ``training_procedure/prepare.py``, we config ``optimizer``, ``loss function`` and ``model init parameters``.

In ``training_procedure/train.py``,  we train the model.

In ``training_procedure/evaluate.py``, we test the model.

## Acknowledgements

This project is inspired by the project [TWIRLS](https://github.com/FFTYYY/TWIRLS) and [IFM_Lab_Program_Template](http://www.ifmlab.org/files/template/IFM_Lab_Program_Template_Python3.zip)
