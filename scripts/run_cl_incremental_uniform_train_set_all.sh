#!/bin/bash

wandb login a88c2cbfefd2a1971a88095e03b1c4f91bce0c47

WANDB_DIR=/home/azad/exp-robopianist/wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
    python /home/azad/robopianist-cl/train-cl.py \
    --root-dir /home/azad/exp-robopianist/ \
    --project robopianist \
    --warmstart-steps 5000 \
    --curriculum incremental_uniform \
    --alpha 1 \
    --segment_core_length 5 \
    --max-steps 6000000 \
    --mode "online" \
    --name "incrmental-uniform-train-set-all" \
    --train_environment_names "train_set_all" \
    --test_environment_names "etude_12"\
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar