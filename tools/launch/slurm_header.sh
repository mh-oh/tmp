#!/bin/bash

source ~/condasetup
conda activate rldev

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_IGNORE_GLOBS="*.nosync"
export PYTHONPATH="/home/minhyeonoh/repos/rldev:$PYTHONPATH"
export PYTHONPATH="/home/minhyeonoh/repos/rldev/stable-baselines3:$PYTHONPATH"