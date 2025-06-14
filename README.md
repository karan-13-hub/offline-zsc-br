# Offline off-belief learning with Behavior Cloning loss

## Introduction

This repo contains the implementation of the algorithm proposed
in [Off-Belief Learning, ICML 2021](https://arxiv.org/pdf/2103.04000.pdf).

## Update
[Mar 2022] We added the code for test time [RL search,
NeurIPS 2021](https://arxiv.org/pdf/2109.15316.pdf) and test time
[Belief Fine-tuning,
ICLR 2022](https://openreview.net/pdf?id=ckZY7DGa7FQ) to this repo as
they were built on the same policy and belief training infrastructure
as off-belief learning. Check out `pyhanabi/rl_search.py` for the main
entry point of the algorithm and `searchcc/` for the backend code that
implements search. Meanwhile, we included a simple single agent search
baseline originally proposed in [SPARTA, AAAI
2020](https://arxiv.org/abs/1912.02318), which can be accessed by
running `pyhanabi/sparta.py`. Check `pyhanabi/README` for more detailed
instructions.

[Feb 2022] We added new code in `pyhanabi/bot` to facilitate playing with
the bot online. Checkout the `pyhanabi/bot/README` for more details.

[Feb 2022] We fixed a major pybind compatibility problem that has been
preventing us from using newer pytorch version. Check the Environment Setup
for more detail.

## Environment Setup
We use conda/miniconda to manage environments.

```shell
conda create -n hanabi python=3.7
conda activate hanabi

pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1

# install other dependencies
pip install psutil

# install a newer cmake if the current version is < 3.15
conda install -c conda-forge cmake
```

To help cmake find the proper libraries (e.g. libtorch), please either
add the following lines to your `.bashrc`, or add it to a separate file
and `source` it before you start working on the project.

```shell
# activate the conda environment
conda activate hanabi

# set path
CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CPATH=${CONDA_PREFIX}/include:${CPATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# avoid tensor operation using all cpu cores
export OMP_NUM_THREADS=1
```

The pybind here works with the pytorch1.10, the latest at the time of writing.
If you use a *newer version of pytorch* that uses a different version
of pybind, first check out the pybind module to use the corresponding
version (the version can be found at pybind11 row
[here](https://github.com/pytorch/pytorch/tree/master/third_party)):
```
cd third_party/pybind11
git checkout $VERSION.XXX
cd ../..
```

Finally, to compile this repo:

```shell
# under project root
mkdir build
cd build
cmake ..
make -j10
```

## Code Structure

For an overview of the training infrastructure, please refer to Figure 5 of the
[Off-Belief Learning] (https://arxiv.org/pdf/2103.04000.pdf) paper.

`hanabi-learning-environment` is a modified version of the original
[HLE from Deepmind](https://github.com/deepmind/hanabi-learning-environment).

Notable modifications includes:

1) Card knowledge part of the observation encoding is changed to
v0-belief, i.e.  card knowledge normalized by the remaining public
card count.

2) Functions to reset the game state with sampled hands.

`rela` (REinforcement Learning Assemly) is a set of tools for
efficient batched neural network inference written in C++ with
multi-threading.

`rlcc` implements the core of various algorithms. For example, the
logic of fictitious transitions are implemented in `r2d2_actor.cc`.
It also contains implementations of baselines such as other-play, VDN
and IQL.

`pyhanabi` is the main entry point of the repo. It contains implementations for
Q-network, recurrent DQN training, belief network and training, as well as some tools
to analyze trained models.

## Run the Code

Please refer to the README in pyhanabi for detailed instruction on how to train a model.

## Download Models

To download the trained models used in the paper, go to `models` folder and run

```shell
sh download.sh
```

Due to agreement with BoardGameArena and Facebook policies, we are
unable to release the "Clone Bot" models trained on the game data nor
the datasets themselves.

## Copyright
Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

This source code is licensed under the license found in the LICENSE
file in the root directory of this source tree.
# offline-zsc-br
