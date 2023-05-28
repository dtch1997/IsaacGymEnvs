#!/bin/bash

GROUP="mania"
for motionIdx in 3 6 8
do
    for temperature in 10.0 1.0 0.1
    do
        python train.py \
            task=QuadrupedAMP \
            task.env.motionFile=data/motions/quadruped/mania/motion$motionIdx.txt \
            headless=True \
            wandb_entity=dtch1997 \
            wandb_project=QuadrupedASE \
            wandb_activate=True \
            wandb_group=$GROUP \
            max_iterations=2000 \
            wandb_name=mania${motionIdx}_temp${temperature} \
            train.params.config.disc_temperature=$temperature
    done
done
