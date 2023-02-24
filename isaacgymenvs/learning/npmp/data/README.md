
`dataset_small.pth`: Approx 50k episodes, each lasting 200 timesteps = 4s, initialized 80% of the time from random states in `dataset_locomotion_recovery.yaml` and the other 20% of the time from a random pose. 

Training command:
```
python train.py test=True num_envs=4096 headless=True checkpoint=data/checkpoints/QuadrupedAMP/QuadrupedAMPGetup/getup_locomotion_randomized.pth task.env.motionFile=data/motions/quadruped/a1_expert/dataset_locomotion_recovery.yaml task.env.enableEarlyTermination=False task.env.enableRefStateInitHeight=True task.env.stateInit=Hybrid task.env.hybridInitProb=0.2 task=QuadrupedAMP task.env.episodeLength_s=4 train.params.config.player.determenistic=False task.env.logging.enableTensorLogging=True train.params.config.player.games_num=50000
```
