# ``rldev``
Reinforcement Learning Algorithms for Development

## Installation

We recommend Anaconda for users for easier installation of Python packages and required libraries.
For a quick start you can simply type the commands below.

```console
$ git clone https://github.com/mh-oh/rldev.git
$ cd rldev
$ conda create -n rldev python=3.8
$ conda activate rldev
$ pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install gym==0.17.1
$ pip install scikit-learn
$ pip install tensorboard
$ pip install tabulate
$ pip install dill
$ pip install "cython<3"
$ pip install mujoco_py
$ pip install numpy==1.23.4
$ pip install overrides
$ pip install scikit-image
$ pip install wandb
$ pip install munch
$ pip install gymnasium
$ pip install gymnasium-robotics
$ pip install ml_collections
$ pip install gpustat
$ pip install moviepy
```

``rldev`` requires you to have Mujoco binaries and a license key.
See [here](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key).

## Algorithms

* [DDPG](https://arxiv.org/abs/1509.02971), [SAC](https://arxiv.org/abs/1812.05905)
* [HER](https://arxiv.org/abs/1802.09464)
* [PEBBLE](https://arxiv.org/abs/2106.05091)

DDPG and HER are based on the version introduced by OpenAI ``baselines`` ([paper](https://arxiv.org/abs/1802.09464), [github](https://github.com/openai/baselines)).

## Getting started

Every experiments are tracked by https://wandb.ai/rldev/experiments.

We exclude large files from wandb cloud synchronization.
To to this, run the below command before running experiments.

```console
$ export WANDB_IGNORE_GLOBS="*.nosync"
```

By doing so, every files with ``.nosync`` as suffix will not be synchronized.

### DDPG and DDPG+HER
```console
$ python experiments/ddpg.py ddpg --env=<environment> --num_envs=8 --seed=1
$ python experiments/ddpg+her.py ddpg+her --env=<environment> --num_envs=8 --seed=1
```
* ``<environment>`` Use one of the followings:
  * fetch-push :heavy_check_mark:
  * fetch-push-dense :heavy_check_mark:
  * fetch-reach :heavy_check_mark:
  * fetch-reach-dense :heavy_check_mark:
  * fetch-pick-and-place :heavy_check_mark:
  * fetch-pick-and-place-dense :heavy_check_mark:
  * point-maze-u
  * point-maze-u-dense
  * point-maze-o
  * point-maze-o-dense

### PEBBLE
```console
$ python experiments/pebble.py <conf> --env=<environment> --num_envs=1 --seed=1
```
* ``<conf>`` Use 'uniform' or any other registered in ``rldev/configs/registry/pebble.py``.
* ``<environment>`` Run ``list_envs.py`` and choose one from the outputs.

## Using launcher to run multiple experiments at once

Make a text file, say ``experiments.txt``, with the following contents.
```
python experiments/ddpg+her.py ddpg+her --env=fetch-push --num_envs=8 --seed=1
python experiments/ddpg+her.py ddpg+her --env=fetch-push --num_envs=8 --seed=2
python experiments/ddpg+her.py ddpg+her --env=fetch-push --num_envs=8 --seed=3
```

Then, execute:
```console
$ python launcher.py experiments.txt --me=<user>
```
- ``<user>`` must be your username in the local machine where the above command runs.

After that, you will see some prompts that you should properly follow.

- You must properly change the contents of ``launcher_header.sh``.
- Add commands that should be excecuted before running each command in ``experiments.txt``.
- Do not make tmux sessions with names starting with ``__autogen__``.

## Visualization

See ``plot.py`` or simply run it:
```console
$ python plot.py
```

## Todo

- [ ] In maze environments, should the reward model be based only on states (not on both states and actions)?
- [x] Using goal-aligned queries rarely fails completely upon random sampling.
- [ ] Traking episodic (pseudo) returns during PEBBLE training will raise error if we use >1 envrionments.
- [x] Noisy samples are given the label -1 by DBSCAN. Simply excluding -1 suffice?
- [ ] By making all experiments use wandb as logging backend, training metrics (e.g., loss, weights, etc.) are currently not tracked.
- [ ] DDPG and DDPG+HER cannot learn anything on maze environments when using 'ddpg-her' config. 
- [ ] In ``plot.py``, implement ``xmax`` parameter for ``curve`` and ``curve_reduce``.
- [x] Goal-aligned entropy sampling for PEBBLE.
- [ ] In DDPG, ``gradient_steps == 1``.
- [ ] KMEANS clustering for goal-aligned queries.
- [ ] For DBSCAN, what ``eps`` parameters to use for each of the following envrionments?
  - point-maze-o
  - point-maze-o-3
  - point-maze-medium
  - button-press

## :x: Deprecated

### DDPG+HER with sparse rewards on FetchPush-v1
```console
$ python experiments/fetch/ddpg+her.py --run=fetch-push.seed=1 --env=FetchPush-v1 --num_envs=8 --seed=1
```

### DDPG+HER with dense rewards on PointMaze_UMaze-v3
```console
$ python experiments/maze/ddpg+her.py --run=u-maze-dense.seed=1 --env=PointMaze_UMazeDense-v3 --num_envs=8 --seed=1
```

### DDPG+HER on button-press-v2
```console
python experiments/metaworld/ddpg+her.py --run=button-press.seed=1 --env=button-press-v2 --num_envs=8 --seed=1
```

### PEBBLE on button-press-v2
```console
$ python experiments/metaworld/pebble.py
```

### PEBBLE on FetchPush-v1
```console
$ python experiments/fetch/pebble.py
```

## References

This code extends and/or modifies the following codebases:

* [Modular RL](https://github.com/spitis/mrl)
* [B-Pref](https://github.com/rll-research/BPref)