#!/bin/bash

GROUP=${1:-"mania_2"}
for motionIdx in 1 2 3 4 6 7 8
do
    python train.py \
        task=QuadrupedAMP \
        task.env.motionFile=data/motions/quadruped/mania/motion$motionIdx.txt \
        task.env.urdfAsset.filepath=urdf/a1_mania.urdf \
        task.env.enableEarlyTermination=True \
        task.env.enableRefStateInitHeight=True \
        train.params.network.space.continuous.learn_sigma=True \
        headless=True \
        wandb_entity=dtch1997 \
        wandb_project=QuadrupedASE \
        wandb_activate=True \
        wandb_group=$GROUP \
        max_iterations=5000 \
        wandb_name=mania${motionIdx}
done
