<p align="center" style="text-align: center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://i.imgur.com/i2NhVg1.png">
  <source media="(prefers-color-scheme: light)" srcset="https://i.imgur.com/9GZNe7m.png">
  <img alt="Neural Language Model" src="https://i.imgur.com/Epg03zu.png">
</picture>
</p>

<p align="center" style="text-align: center">
    Neural Language Model
</p>

<p align="center" style="text-align: center">
    <a href="https://stand-with-ukraine.pp.ua">
        <img
            src="https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/badges/StandWithUkraine.svg"
            alt="Stand With Ukraine"
        />
    </a>
</p>

<p align="center" style="text-align: center">
    <a
        href="https://github.com/vivekjoshy/OpenGrammar/actions/workflows/main.yml">
            <img
                src="https://github.com/vivekjoshy/OpenGrammar/actions/workflows/main.yml/badge.svg"
                alt="Tests"
    />
    </a>
    <a
        href="https://codecov.io/gh/vivekjoshy/OpenGrammar">
            <img
                src="https://codecov.io/gh/vivekjoshy/OpenGrammar/branch/main/graph/badge.svg?token=Ep07QEelsi"
                alt="codecov" />
    </a>
    <img src="https://img.shields.io/pypi/dm/opengrammar"
        alt="PyPI - Downloads"
    />
    <a
        href="https://opengrammar.rtfd.io/en/latest/?badge=latest">
            <img
                src="https://readthedocs.org/projects/opengrammar/badge/?version=latest"
                    alt="Documentation Status"
            />
    </a>
</p>

## Description

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opengrammar)
![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)

Open Grammar is a biologically inspired language model. 

## Installation
```shell
pip install opengrammar
```

## Usage


### Training
To start training the model, you need to set the path to the minipile dataset.

You can do this by either defining a `config.toml` file as seen below or setting an
`OG_MINIPILE_ROOT` environment variable.

Then simply run one of the following commands:

```sh
opengrammar train
```

OR

```sh
opengrammar train --config /path/to/config.toml
```

Environment variables which will take precedence over `config.toml` files.
For instance you can set the token environment like so:

```sh
export OG_WANDB_TOKEN=token_goes_here
```

# Configuration


| Config          | Type   | Default   | Description                                            |
|-----------------|--------|-----------|--------------------------------------------------------|
| `minipile_root` | `Path` | Not Set   | A path to the Minipile dataset folder. [Mandatory]     |
| `wandb_token`   | `str`  | Not Set   | An API key for WandB. [Optional]                       |
| `batch`         | `bool` | `4`       | The number of samples per batch.                       |
| `lr`            | `int`  | `0.00001` | The learning rate.                                     |
| `epochs`        | `int`  | `10`      | The number of epochs to train for.                     |
| `hidden_size`   | `int`  | `128`     | The number of hidden dimensions used for model layers. |
| `tensor_cores`  | `bool` | `true`    | Enable or disable usage of tensor cores in your GPU.   |
| `devices`       | `int`  | `1`       | The number of GPUs available to use.                   |
| `random_seed`   | `int`  | `7`       | A random seed for reproducible training.               |
| `debug`         | `bool` | `False`   | Disables expensive code and prints debugging logs.     |

### Sample Config

```toml
minipile_root = "resources/minipile"
wandb_token = "very_secret_token"
batch = 4
lr = 0.00001
epochs = 10
hidden_dim = 16
tensor_cores = true
devices = 1
random_seed = 7
debug = false
```
