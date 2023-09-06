#!/bin/bash
#export TMPDIR=/home/azad/robo-tmp
#export WANDB_CONFIG_DIR=/home/azad/robo-tmp
#export WANDB_CACHE_DIR=/home/azad/robo-tmp
wandb login a88c2cbfefd2a1971a88095e03b1c4f91bce0c47

WANDB_DIR=/home/azad/robo-exp/wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir /home/azad/robo-exp/ \
    --warmstart-steps 5000 \
    --max-steps 1000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --mode "online" \
    --tqdm-bar

WANDB_DIR=/home/azad/robo-exp/wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir /home/azad/robo-exp/ \
    --warmstart-steps 5000 \
    --max-steps 1000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-etude-12-FrenchSuiteNo5Sarabande-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --mode "online" \
    --tqdm-bar


WANDB_DIR=/home/azad/robo-exp/wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir /home/azad/robo-exp/ \
    --warmstart-steps 5000 \
    --max-steps 1000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-etude-12-PianoSonataD8451StMov-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --mode "online" \
    --tqdm-bar

WANDB_DIR=/home/azad/robo-exp/wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir /home/azad/robo-exp/ \
    --warmstart-steps 5000 \
    --max-steps 1000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-etude-12-PartitaNo26-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --mode "online" \
    --tqdm-bar

WANDB_DIR=/home/azad/robo-exp/wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir /home/azad/robo-exp/ \
    --warmstart-steps 5000 \
    --max-steps 1000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-etude-12-WaltzOp64No1-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --mode "online" \
    --tqdm-bar

WANDB_DIR=/home/azad/robo-exp/wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir /home/azad/robo-exp/ \
    --warmstart-steps 5000 \
    --max-steps 1000000 \
    --discount 0.8 \
    --agent-config.critic-dropout-rate 0.01 \
    --agent-config.critic-layer-norm \
    --agent-config.hidden-dims 256 256 256 \
    --trim-silence \
    --gravity-compensation \
    --reduced-action-space \
    --control-timestep 0.05 \
    --n-steps-lookahead 10 \
    --environment-name "RoboPianist-etude-12-BagatelleOp3No4-v0" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --mode "online" \
    --tqdm-bar