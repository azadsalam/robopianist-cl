#!/bin/bash

wandb login a88c2cbfefd2a1971a88095e03b1c4f91bce0c47

WANDB_DIR=/home/azad/robo-exp/wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train-cl.py \
    --root-dir /home/azad/robo-exp/ \
    --train_environment_names etude_12 \
    --test_environment_names etude_12 \
    --project robopianist-debug \
    --warmstart-steps 400 \
    --curriculum inverse_reward \
    --alpha 1 \
    --segment_core_length 50 \
    --overlap_left 0.1 \
    --overlap_right 0.1 \
    --max-steps 2000 \
    --log_interval 200 \
    --eval_interval 500 \
    --mode "online" \
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
