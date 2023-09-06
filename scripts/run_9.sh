#!/bin/bash

# List of environment values
environments=( 'RoboPianist-etude-12-PartitaNo26-v0'
    'RoboPianist-etude-12-WaltzOp64No1-v0' 'RoboPianist-etude-12-BagatelleOp3No4-v0'
    'RoboPianist-etude-12-KreislerianaOp16No8-v0' 'RoboPianist-etude-12-FrenchSuiteNo5Gavotte-v0'
    'RoboPianist-etude-12-PianoSonataNo232NdMov-v0' 'RoboPianist-etude-12-GolliwoggsCakewalk-v0'
    'RoboPianist-etude-12-PianoSonataNo21StMov-v0' 'RoboPianist-etude-12-PianoSonataK279InCMajor1StMov-v0')

# Iterate over each environment value
for env in "${environments[@]}"; do
    echo "Running with environment: $env"
    WANDB_DIR=/home/azad/exp-robopianist/wandb MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python train.py \
    --root-dir /home/azad/exp-robopianist/ \
    --project robopianist \
    --warmstart-steps 5000 \
    --max-steps 1000000 \
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
    --environment-name "$env" \
    --action-reward-observation \
    --primitive-fingertip-collisions \
    --eval-episodes 1 \
    --camera-id "piano/back" \
    --tqdm-bar
done