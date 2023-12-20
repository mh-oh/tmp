# ``rldev``
Reinforcement Learning Algorithms for Development

## Installation

We recommend Anaconda for users for easier installation of Python packages and required libraries.
For a quick start you can simply type the commands below.

```console
~$ git clone https://github.com/mh-oh/rldev.git
~$ cd rldev
~$ conda create -n rldev python=3.8
~$ conda activate
~$ pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
~$ pip install gym==0.17.1
~$ pip install scikit-learn
~$ pip install tensorboard
~$ pip install tabulate
~$ pip install dill
~$ pip install "cython<3"
~$ pip install mujoco_py
~$ pip install numpy==1.23.4
```

``rldev`` requires you to have Mujoco binaries and a license key.
See [here](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key).

## Algorithms

* [DDPG](https://arxiv.org/abs/1509.02971), [SAC](https://arxiv.org/abs/1812.05905)
* [HER](https://arxiv.org/abs/1802.09464)
* [PEBBLE](https://arxiv.org/abs/2106.05091)

DDPG and HER are based on the version introduced by OpenAI ``baselines`` ([paper](https://arxiv.org/abs/1802.09464), [github](https://github.com/openai/baselines)).

## Getting started

```console
~$ python experiments/fetch/ddpg+her.py --run=fetchpush.seed=1 --env=FetchPush-v1 --num_envs=8 --seed=1
```
